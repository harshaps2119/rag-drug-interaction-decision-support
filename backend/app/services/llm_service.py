"""
services/llm_service.py
==========================
Talks to xAI Grok (via the OpenAI-compatible API), OpenRouter-hosted models
(also OpenAI-compatible), or Google Gemini to generate a STRUCTURED,
evidence-grounded explanation of a drug pair's retrieved evidence — never to
determine whether an interaction exists. See the module docstring in
services/explanation_service.py for how this fits into the full pipeline,
and docs/llm.md for the full safety reasoning.

SUPPORTED PROVIDERS
--------------------
  xai        — xAI Grok via XAILLMClient (OpenAI-compatible, strict JSON schema)
  openrouter — Any OpenRouter model via OpenRouterLLMClient (OpenAI-compatible)
  gemini     — Google Gemini via GeminiLLMClient (google-generativeai SDK)
  none       — DisabledLLMClient (always raises, triggers evidence-only fallback)

WHY THE PROVIDER IS SWAPPABLE (LLMClient protocol)
--------------------------------------------------------
`generate_grounded_response()` accepts any object with a `.generate(prompt) -> str`
method — it doesn't import or know about a provider specifically.
XAILLMClient and GeminiLLMClient are small provider implementations;
adding another provider does not change the prompt, schema, retrieval,
or grounding validator.

WHY xAI USES THE OPENAI-COMPATIBLE SDK
--------------------------------------
xAI documents an OpenAI-compatible endpoint at https://api.x.ai/v1. The
xAI client uses strict JSON Schema output generated from
LLMStructuredOutput. This improves response format reliability but does
not replace the application's independent parsing or grounding controls.

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
Read from the selected provider's settings (from `.env` via app/config.py)
— NEVER hard-coded, and never logged. If its key is empty,
LLMAPIKeyMissingError is raised before any network call is attempted, so the caller
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
    LLMServiceError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from app.schemas.evidence_assessment import PairEvidenceItem
from app.schemas.llm_output import LLMStructuredOutput

MAX_EVIDENCE_TEXT_CHARS = 1200  # defensive cap per evidence item's text in the prompt, to keep prompts bounded


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class XAILLMClient:
    """xAI Grok implementation using xAI's OpenAI-compatible API."""

    source = "xAI"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        sdk_client=None,
    ):
        self.api_key = api_key if api_key is not None else settings.XAI_API_KEY
        self.model_name = model_name or settings.XAI_MODEL
        self.base_url = base_url or settings.XAI_BASE_URL
        self.timeout_seconds = timeout_seconds or settings.LLM_TIMEOUT_SECONDS
        # Dependency injection keeps provider tests offline and avoids
        # importing the SDK until a real xAI request is needed.
        self._client = sdk_client

    def _load(self):
        if not self.api_key:
            raise LLMAPIKeyMissingError(
                "XAI_API_KEY is not set. Add it to your .env file — see .env.example.", source="xAI"
            )
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMServiceUnavailableError(
                    "The xAI client dependency is not installed. Install backend/requirements.txt.",
                    source="xAI",
                ) from exc

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                # Let the existing evidence-only fallback handle a failed
                # explanation promptly rather than making hidden retries.
                max_retries=0,
            )
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._load()
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ddi_grounded_explanation",
                        "strict": True,
                        "schema": LLMStructuredOutput.model_json_schema(),
                    },
                },
            )
        except Exception as exc:
            raise _translate_xai_exception(exc) from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMOutputParsingError("xAI returned a response without a message body.", source="xAI") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMOutputParsingError("xAI returned an empty response.", source="xAI")
        return content


class GeminiLLMClient:
    """The real Gemini-backed implementation of LLMClient."""

    source = "Gemini"

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
                "GEMINI_API_KEY is not set. Add it to your .env file — see .env.example.", source="Gemini"
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
            raise LLMOutputParsingError("Gemini returned an empty response.", source="Gemini")
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
        return LLMAPIKeyInvalidError(f"Gemini rejected the API key: {message}", source="Gemini")
    if any(k in haystack for k in ("deadlineexceeded", "timeout", "timed out")):
        return LLMTimeoutError(f"Gemini request timed out: {message}", source="Gemini")
    if any(k in haystack for k in ("resourceexhausted", "rate limit", "429", "quota")):
        return LLMRateLimitError(f"Gemini rate limit hit: {message}", source="Gemini")
    if any(k in haystack for k in ("serviceunavailable", "internalservererror", "503", "500")):
        return LLMServiceUnavailableError(f"Gemini service unavailable: {message}", source="Gemini")
    return LLMServiceUnavailableError(f"Gemini call failed: {message}", source="Gemini")


def _translate_xai_exception(exc: Exception) -> LLMServiceError:
    """Translate OpenAI-compatible xAI client failures into safe domain errors."""
    name = type(exc).__name__
    message = str(exc)
    haystack = f"{name} {message}".lower()
    status_code = getattr(exc, "status_code", None)

    if status_code in (401, 403) or any(
        marker in haystack
        for marker in (
            "authentication",
            "unauthorized",
            "invalid api key",
            "invalid_api_key",
            "incorrect api key",
        )
    ):
        return LLMAPIKeyInvalidError(f"xAI rejected the API key: {message}", source="xAI")
    if status_code in (408, 504) or any(marker in haystack for marker in ("timeout", "timed out", "apitimeouterror")):
        return LLMTimeoutError(f"xAI request timed out: {message}", source="xAI")
    if status_code == 429 or any(marker in haystack for marker in ("rate limit", "ratelimit", "429", "quota")):
        return LLMRateLimitError(f"xAI rate limit hit: {message}", source="xAI")
    if (isinstance(status_code, int) and status_code >= 500) or any(
        marker in haystack for marker in ("connection", "api connection", "service unavailable", "503", "500")
    ):
        return LLMServiceUnavailableError(f"xAI service unavailable: {message}", source="xAI")
    return LLMServiceUnavailableError(f"xAI call failed: {message}", source="xAI")


def _translate_openrouter_exception(exc: Exception) -> LLMServiceError:
    """Translate OpenAI-compatible OpenRouter client failures into safe domain errors."""
    name = type(exc).__name__
    message = str(exc)
    haystack = f"{name} {message}".lower()
    status_code = getattr(exc, "status_code", None)

    if status_code in (401, 403) or any(
        marker in haystack
        for marker in (
            "authentication",
            "unauthorized",
            "invalid api key",
            "invalid_api_key",
            "no auth",
        )
    ):
        return LLMAPIKeyInvalidError(f"OpenRouter rejected the API key: {message}", source="OpenRouter")
    if status_code in (408, 504) or any(marker in haystack for marker in ("timeout", "timed out", "apitimeouterror")):
        return LLMTimeoutError(f"OpenRouter request timed out: {message}", source="OpenRouter")
    if status_code == 429 or any(marker in haystack for marker in ("rate limit", "ratelimit", "429", "quota")):
        return LLMRateLimitError(f"OpenRouter rate limit hit: {message}", source="OpenRouter")
    if (isinstance(status_code, int) and status_code >= 500) or any(
        marker in haystack for marker in ("connection", "api connection", "service unavailable", "503", "500")
    ):
        return LLMServiceUnavailableError(f"OpenRouter service unavailable: {message}", source="OpenRouter")
    return LLMServiceUnavailableError(f"OpenRouter call failed: {message}", source="OpenRouter")


class DisabledLLMClient:
    """A client used when LLM explanations are disabled in configuration."""

    source = "disabled"

    def generate(self, prompt: str) -> str:
        raise LLMAPIKeyMissingError("LLM provider is disabled.", source="disabled")


class OpenRouterLLMClient:
    """
    OpenRouter implementation using their OpenAI-compatible REST API.

    OpenRouter proxies many hosted models (Llama, Mistral, Claude, GPT-4, etc.)
    through a single endpoint with a standard OpenAI-SDK interface. We reuse
    the same transport pattern as XAILLMClient, adding the two headers
    required by OpenRouter's routing layer.

    WHY NO STRICT JSON_SCHEMA MODE
    --------------------------------
    OpenRouter passes requests through to the upstream model provider, and
    many upstream providers do not support the ``response_format={"type":
    "json_schema", ...}`` parameter. We therefore request
    ``{"type": "json_object"}`` (supported broadly) and rely on the prompt's
    explicit JSON-only instruction plus the existing ``_strip_code_fences``
    / ``generate_grounded_response`` parsing/validation layer, which was
    already hardened for models that wrap JSON in markdown fences.

    HOW THE API KEY IS HANDLED
    ---------------------------
    Read from settings.OPENROUTER_API_KEY (from .env via config.py) — never
    hard-coded, never logged. If the key is empty, LLMAPIKeyMissingError is
    raised before any network call so explanation_service falls back to
    evidence-only mode immediately.
    """

    source = "OpenRouter"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        sdk_client=None,
    ):
        self.api_key = api_key if api_key is not None else settings.OPENROUTER_API_KEY
        self.model_name = model_name or settings.OPENROUTER_MODEL
        self.base_url = base_url or settings.OPENROUTER_BASE_URL
        self.timeout_seconds = timeout_seconds or settings.LLM_TIMEOUT_SECONDS
        self._client = sdk_client

    def _load(self):
        if not self.api_key:
            raise LLMAPIKeyMissingError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file — see .env.example.",
                source="OpenRouter",
            )
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMServiceUnavailableError(
                    "The openai package is not installed. Install backend/requirements.txt.",
                    source="OpenRouter",
                ) from exc

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                max_retries=0,
                # OpenRouter routing layer requires these two headers.
                default_headers={
                    "HTTP-Referer": "https://github.com/your-org/rag-ddi",
                    "X-Title": "RAG DDI Decision Support",
                },
            )
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._load()
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                # Use json_object mode (broadly supported by OpenRouter models)
                # rather than strict json_schema (not universally supported).
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise _translate_openrouter_exception(exc) from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMOutputParsingError(
                "OpenRouter returned a response without a message body.", source="OpenRouter"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMOutputParsingError("OpenRouter returned an empty response.", source="OpenRouter")
        return content


def get_configured_llm_client(provider: str | None = None) -> LLMClient:
    """Return the selected explanation client; retrieval and validation stay provider-independent."""
    selected_provider = (provider or settings.LLM_PROVIDER).strip().lower()
    if selected_provider == "xai":
        return XAILLMClient()
    if selected_provider == "gemini":
        return GeminiLLMClient()
    if selected_provider == "openrouter":
        return OpenRouterLLMClient()
    if selected_provider in ("none", "disabled", "off"):
        return DisabledLLMClient()
    # Settings validates deployed values, but keep direct calls and monkeypatches safe.
    raise LLMServiceUnavailableError(
        f"Unsupported LLM_PROVIDER '{selected_provider}'. Use 'xai', 'gemini', 'openrouter', or 'none'.",
        source="configuration",
    )


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
    client = client or get_configured_llm_client()
    source = getattr(client, "source", "LLM")
    prompt = build_prompt(drug_a_name, drug_b_name, evidence_map)

    raw_text = client.generate(prompt)

    cleaned = _strip_code_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMOutputParsingError(f"Model did not return valid JSON: {exc}", source=source) from exc

    try:
        return LLMStructuredOutput(**data)
    except Exception as exc:  # pydantic.ValidationError, or a non-dict payload
        raise LLMOutputParsingError(f"Model's JSON did not match the expected schema: {exc}", source=source) from exc


def _strip_code_fences(text: str) -> str:
    """Defensive: some models wrap JSON in ```json ... ``` even when asked not to. Strip it if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    return stripped.strip()
