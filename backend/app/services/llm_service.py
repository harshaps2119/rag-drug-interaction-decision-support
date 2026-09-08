"""
services/llm_service.py
==========================
Talks to Gemini (or, via the LLMClient protocol, any swapped-in
provider) to generate a STRUCTURED, evidence-grounded explanation of a
drug pair's retrieved evidence — never to determine whether an
interaction exists. See the module docstring in
services/explanation_service.py for how this fits into the full
pipeline, and docs/llm.md for the full safety reasoning.

WHY THE PROVIDER IS SWAPPABLE (LLMClient protocol)
--------------------------------------------------------
`generate_grounded_response()` accepts any object with a `.generate(prompt) -> str`
method — it doesn't import or know about Gemini specifically. GeminiLLMClient
is the only implementation right now, but swapping in OpenAI or Claude
later means writing one new small class, not touching this file's logic,
the prompt, the schema, or the validator at all.

WHY REST TRANSPORT, NOT THE SDK'S DEFAULT (gRPC)
------------------------------------------------------
Discovered by actually testing this in the project's build sandbox: the
Gemini SDK's default gRPC transport does NOT reliably respect the
`request_options={"timeout": ...}` parameter when the network is
restricted — a direct test here hung past its 15-second bash-level
timeout and had to be killed, rather than raising a timeout error as
expected. Passing `transport="rest"` to `genai.configure()` uses plain
HTTPS instead, which respects request timeouts properly. This matters
for real deployments too (e.g. behind a restrictive corporate proxy),
not just this sandbox.

HOW THE API KEY IS HANDLED
-------------------------------
Read from `settings.GEMINI_API_KEY` (from `.env` via app/config.py) —
NEVER hard-coded, and never logged. If it's empty, LLMAPIKeyMissingError
is raised before any network call is attempted, so the caller
(explanation_service) can fall back to evidence-only mode immediately
rather than waiting on a network call doomed to fail.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_llm_service.py -v

These tests use a FakeLLMClient (see tests/_fake_llm.py) implementing
the same `.generate()` interface — they test PROMPT CONTENT, RESPONSE
PARSING, and EXCEPTION TRANSLATION without ever calling the real Gemini
API. For an actual live call, see scripts/test_gemini_live.py — NOT
executed by Claude (see that script's docstring for why).
"""

from __future__ import annotations

import json
from typing import Protocol

from app.config import settings
from app.exceptions import (
    LLMAPIKeyInvalidError,
    LLMAPIKeyMissingError,
    LLMOutputParsingError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from app.schemas.evidence_assessment import PairEvidenceItem
from app.schemas.llm_output import LLMStructuredOutput

MAX_EVIDENCE_TEXT_CHARS = 1200  # defensive cap per evidence item's text in the prompt, to keep prompts bounded


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class GeminiLLMClient:
    """The real Gemini-backed implementation of LLMClient."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self.timeout_seconds = timeout_seconds or settings.LLM_TIMEOUT_SECONDS
        self._model = None

    def _load(self):
        if not self.api_key:
            raise LLMAPIKeyMissingError(
                "GEMINI_API_KEY is not set. Add it to your .env file — see .env.example."
            )
        if self._model is None:
            import google.generativeai as genai

            # transport="rest" — see module docstring for why this matters for timeouts.
            genai.configure(api_key=self.api_key, transport="rest")
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def generate(self, prompt: str) -> str:
        model = self._load()
        try:
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json", "temperature": 0.0},
                request_options={"timeout": self.timeout_seconds},
            )
        except Exception as exc:
            raise _translate_gemini_exception(exc) from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMOutputParsingError("Gemini returned an empty response.")
        return text


def _translate_gemini_exception(exc: Exception) -> Exception:
    """
    Maps whatever the Gemini SDK/google.api_core raises into this
    project's own exception types, so callers never need to know about
    Gemini/gRPC/google.api_core internals. Matched by class name and
    message content rather than importing google.api_core's exception
    classes directly, so this file has no hard dependency on exactly
    which exception hierarchy the SDK uses internally (that has changed
    across SDK versions).
    """
    name = type(exc).__name__
    message = str(exc)
    haystack = f"{name} {message}".lower()

    if any(k in haystack for k in ("unauthenticated", "permissiondenied", "api_key_invalid", "invalid api key")):
        return LLMAPIKeyInvalidError(f"Gemini rejected the API key: {message}")
    if any(k in haystack for k in ("deadlineexceeded", "timeout", "timed out")):
        return LLMTimeoutError(f"Gemini request timed out: {message}")
    if any(k in haystack for k in ("resourceexhausted", "rate limit", "429", "quota")):
        return LLMRateLimitError(f"Gemini rate limit hit: {message}")
    if any(k in haystack for k in ("serviceunavailable", "internalservererror", "503", "500")):
        return LLMServiceUnavailableError(f"Gemini service unavailable: {message}")
    return LLMServiceUnavailableError(f"Gemini call failed: {message}")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTIONS = """You are an evidence-grounded clinical information assistant.

You will be given retrieved evidence excerpts from FDA drug labels about two named drugs, Drug A and Drug B. Each excerpt has a stable id (EVIDENCE-001, EVIDENCE-002, ...) and is labeled with how it relates to the pair: "pair_specific" (the excerpt's own source drug's label explicitly names the OTHER drug too), "class_level" (names a drug class the other drug belongs to), "drug_specific" (only about one of the two drugs, no connection to the other), or "general_label" (retrieved but not clearly about either drug).

STRICT RULES — follow every one of these exactly:
1. Use ONLY the supplied evidence below. Do not rely on your own pretrained medical knowledge to fill any gap.
2. If the evidence does not establish a connection between Drug A and Drug B specifically, do not claim that an interaction exists between them.
3. Never state or imply "no interaction exists", "these drugs are safe to combine", or any equivalent — insufficient or supporting-only evidence must be described as insufficient/uncertain, NEVER as proof of safety. Silence in the evidence is not evidence of safety.
4. Do not infer, invent, or estimate a severity. If the evidence explicitly states a severity/risk word (e.g. "contraindicated", "serious", "major"), copy that exact short phrase into the severity field. Otherwise set severity to null. Never map source wording to a different standardized category.
5. Do not invent a mechanism. Only state a mechanism if the evidence explicitly describes one.
6. Do not fabricate citations, references, or URLs of any kind. Refer to evidence ONLY by its EVIDENCE-XXX id, in the cited_evidence_ids field. Never write a URL, DOI, or reference name anywhere in your response.
7. Preserve uncertainty. If the only evidence is class_level or drug_specific (not pair_specific), your evidence_summary must say so explicitly. In that situation, you MUST NOT state or imply that Drug A and Drug B interact, must NOT provide a pair-specific clinical effect or mechanism, and must NOT describe the supporting evidence as proof of an interaction. You may only summarize what the supplied evidence says about the individual drug or class relationship. Do not turn absence of pair-specific evidence into a claim that the drugs do not interact.
8. Your interaction_assessment must never exceed what the evidence actually supports: only use "pair_specific_evidence_found" if at least one EVIDENCE-XXX item is labeled pair_specific; only use "supporting_evidence_found" if there is drug_specific or class_level evidence but no pair_specific evidence; use "insufficient_evidence" if there is nothing relevant at all.
9. Respond with JSON ONLY — no markdown, no code fences, no prose outside the JSON object — matching exactly this shape:
{"interaction_assessment": "pair_specific_evidence_found" | "supporting_evidence_found" | "insufficient_evidence", "evidence_summary": "...", "clinical_effect": "..." or null, "mechanism": "..." or null, "severity": "..." or null, "cited_evidence_ids": ["EVIDENCE-001", ...], "limitations": ["..."], "safety_notice": "..."}
"""


def build_prompt(
    drug_a_name: str,
    drug_b_name: str,
    evidence_map: dict[str, PairEvidenceItem],
) -> str:
    """
    Builds the full prompt: the strict system instructions above, plus
    the drug pair and the labeled evidence excerpts. If evidence_map is
    empty (Phase 6 found nothing at all), the prompt says so explicitly
    rather than omitting the section, so the model can't mistake "no
    evidence block shown" for "go ahead and use your own knowledge".
    """
    lines = [
        _SYSTEM_INSTRUCTIONS,
        "",
        f"Drug A: {drug_a_name}",
        f"Drug B: {drug_b_name}",
        "",
    ]

    if not evidence_map:
        lines.append("EVIDENCE: none retrieved. There is no evidence to summarize.")
    else:
        lines.append("EVIDENCE:")
        for evidence_id, item in evidence_map.items():
            text = item.text[:MAX_EVIDENCE_TEXT_CHARS]
            lines.append(
                f"[{evidence_id}] (relation: {item.classification.value}, source drug: {item.drug_name}, "
                f"section: {item.section_name})\n{text}"
            )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_grounded_response(
    drug_a_name: str,
    drug_b_name: str,
    evidence_map: dict[str, PairEvidenceItem],
    client: LLMClient | None = None,
) -> LLMStructuredOutput:
    """
    Builds the prompt, calls the LLM client, and parses its response into
    LLMStructuredOutput. Raises LLMServiceError subclasses for any
    failure (missing/invalid key, timeout, rate limit, service
    unavailable, malformed output) — callers (explanation_service) catch
    these and fall back to evidence-only mode. Never returns a partially
    -trusted result; parsing/schema failure is always all-or-nothing.
    """
    client = client or GeminiLLMClient()
    prompt = build_prompt(drug_a_name, drug_b_name, evidence_map)

    raw_text = client.generate(prompt)

    cleaned = _strip_code_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMOutputParsingError(f"Model did not return valid JSON: {exc}") from exc

    try:
        return LLMStructuredOutput(**data)
    except Exception as exc:  # pydantic.ValidationError, or a non-dict payload
        raise LLMOutputParsingError(f"Model's JSON did not match the expected schema: {exc}") from exc


def _strip_code_fences(text: str) -> str:
    """Defensive: some models wrap JSON in ```json ... ``` even when asked not to. Strip it if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    return stripped.strip()
