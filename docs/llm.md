# LLM Integration — Phase 7

## Why the LLM is not the source of truth

Every fact in this system — which drug, which FDA label, which section,
which exact text — was already established by Phases 2-6, entirely
without any LLM involvement. xAI Grok is the default provider in this phase for exactly
one job: **turning already-retrieved, already-classified evidence into
readable prose** — never to decide what counts as evidence, never to
decide whether an interaction exists, never to add facts the retrieval
layer didn't already surface.

This is enforced mechanically, not just by asking nicely in the prompt:
`services/grounding_validator.py` runs after every LLM call and
independently checks that the model didn't exceed what Phase 6's
retrieval actually supports (see "Grounding validation" below). If it
did, the response is discarded — the LLM's opinion of what it found is
never trusted over the retrieval layer's own classification.

## Why RAG is used at all (recap, in this phase's context)

Without retrieval, an LLM asked about a drug pair answers from whatever
it learned during training — ungrounded, unverifiable, and potentially
outdated or wrong. RAG (Phases 5-6) means Gemini is only ever asked to
*read and explain specific, real, retrieved FDA label text* — never to
recall facts from its own training. The prompt says this explicitly and
repeatedly (see "The prompt" below), and the validator checks it.

## Provider configuration and structured output

`LLM_PROVIDER=xai` selects the default provider, xAI Grok. The xAI client
uses the official OpenAI-compatible endpoint (`https://api.x.ai/v1`) with
`XAI_MODEL=grok-4.6` and strict JSON Schema output generated from
`LLMStructuredOutput`. Gemini remains available for existing deployments
with `LLM_PROVIDER=gemini`. Both providers feed the same parser and the
same local grounding validator; no provider is allowed to determine the
interaction evidence status.

## How evidence is passed to the LLM

`services/explanation_service.build_evidence_id_map()` takes a Phase 6
`PairRetrievalResult` and assigns each evidence chunk a stable id —
`EVIDENCE-001`, `EVIDENCE-002`, ... — in a fixed order (pair-specific
evidence first, then supporting evidence), capped at
`settings.LLM_MAX_EVIDENCE_ITEMS` (default 8) to keep prompts bounded.

`services/llm_service.build_prompt()` then writes each evidence item
into the prompt labeled with its id, its classification (`pair_specific`
/ `class_level` / `drug_specific` / `general_label`), its source drug,
and its section name — so the LLM has, in plain sight, exactly the
distinction Phase 6 already made, and is instructed not to blur it.

If Phase 6 found nothing at all, the prompt says so explicitly
("EVIDENCE: none retrieved. There is no evidence to summarize.") rather
than silently omitting the evidence section — so the model can't mistake
an empty section for permission to use its own knowledge instead.

## The prompt

The full system instructions live in `services/llm_service.py` as
`_SYSTEM_INSTRUCTIONS`. It opens with the required framing:

> "You are an evidence-grounded clinical information assistant."

and includes, verbatim, every rule the project specification requires:
use only supplied evidence; never rely on pretrained knowledge to fill
gaps; never claim an interaction the evidence doesn't establish; never
infer severity; never fabricate citations; preserve uncertainty;
distinguish pair-specific from drug-specific/class-level evidence. The
prompt also fully specifies the exact JSON shape expected in response,
so parsing failures are rare and unambiguous when they happen.

## Structured output

The provider's raw text response is parsed into `schemas/llm_output.py`'s
`LLMStructuredOutput` — a narrow schema with exactly the fields the
prompt asked for: `interaction_assessment`, `evidence_summary`,
`clinical_effect`, `mechanism`, `severity`, `cited_evidence_ids`,
`limitations`, `safety_notice`. Any JSON that doesn't parse, or doesn't
match this schema (wrong types, an assessment value outside the three
allowed literals, a missing required field), is treated as
`LLMOutputParsingError` — a total failure, routed straight to the
evidence-only fallback. There is no "best effort" partial parsing.

### Why `severity` is nullable and unstandardized

The schema's `severity` field is `str | None` — never an enum. The
project explicitly prohibits mapping source wording ("serious",
"contraindicated") into an invented standardized category. When the
retrieved evidence doesn't state a severity, the correct value is
`null` (surfaced to the user as "Not stated in retrieved source"), and
the prompt instructs Gemini accordingly. When it *does* state one,
Gemini is told to copy the exact short phrase — and the validator then
independently checks that phrase actually appears in the cited evidence
text (see below) before it's ever shown.

## How citations are handled — and why the LLM never generates them itself

Gemini is never allowed to write a URL, DOI, or reference of any kind —
the prompt says so directly, and the validator rejects any response that
does (see "Fabricated URL detection" below). Instead, Gemini can only
refer to evidence by the `EVIDENCE-XXX` ids it was given. The backend —
not the model — maps those ids back to full source detail (DailyMed
label, section, URL, chunk id, original text) via
`explanation_service._citation_from_item()`, using data that was already
verified and stored in Phases 3-6. This is strictly safer than letting
the model generate citations itself: a hallucinated `EVIDENCE-999` id is
mechanically detectable (it simply isn't in the map the backend built),
whereas a hallucinated URL or reference name would look just as
plausible as a real one and be far harder to catch.

## Grounding validation

`services/grounding_validator.validate()` runs after every Gemini call,
before anything is shown to a user. It checks, independently of
anything Gemini claims about itself:

1. **Every cited evidence id actually exists** in what was supplied —
   otherwise: hallucinated citation, fatal.
2. **The claimed `interaction_assessment` never exceeds what Phase 6's
   retrieval actually found** — the model cannot upgrade
   `supporting_evidence_found` into `pair_specific_evidence_found`, or
   `insufficient_evidence` into anything stronger. Fatal if violated.
3. **No "no interaction" language anywhere in the free text** — a
   mechanical phrase check across `evidence_summary`, `clinical_effect`,
   `mechanism`, and `safety_notice`. Fatal if found.
4. **No fabricated URLs** anywhere in the free text. Fatal if found.
5. **Severity must be verbatim-supported** by the cited evidence text.
   This one is *not* fatal — an unsupported severity is stripped to
   `null` and the rest of the (otherwise sound) response is still shown,
   with the correction recorded. See `docs/safety.md` for why this one
   check is handled differently from the other four.

If any of checks 1-4 fail, the entire LLM response is discarded — the
system falls back to evidence-only mode (below). There is no partial
trust: a model willing to hallucinate one citation or upgrade one
classification cannot be trusted on the surrounding claims either.

## What happens when the LLM fails, in any way

Every failure mode below routes to the exact same evidence-only
fallback — full retrieved evidence, with citations, no generated prose,
plus an explicit reason:

| Failure | Exception raised | Where |
|---|---|---|
| No API key configured | `LLMAPIKeyMissingError` | Before any network call |
| API key rejected | `LLMAPIKeyInvalidError` | On the provider call |
| Request times out | `LLMTimeoutError` | On the provider call |
| Rate limit hit | `LLMRateLimitError` | On the provider call |
| Provider service down (5xx) | `LLMServiceUnavailableError` | On the provider call |
| Response isn't valid JSON | `LLMOutputParsingError` | Parsing the response |
| Response JSON doesn't match schema | `LLMOutputParsingError` | Parsing the response |
| Response fails grounding validation | *(no exception — `ValidationResult.passed=False`)* | After parsing, before display |

`services/explanation_service.generate_explanation()` never raises for
any of these — it always returns a valid `ExplanationResponse`, with
`mode="evidence_only"` and a human-readable `fallback_reason` for every
case above. The application stays usable in evidence-only mode
regardless of which failure occurred.

## Provider replaceability

`services/llm_service.py` defines a minimal `LLMClient` protocol
(`.generate(prompt: str) -> str`). `GeminiLLMClient` is the only
implementation today, but nothing outside this one file (and the small
translation function inside it) knows anything about Gemini
specifically. Swapping in OpenAI or Claude later means writing one new
class satisfying that same protocol — the prompt, the schema, the
validator, and the orchestration in `explanation_service.py` all stay
unchanged.

## What was NOT verified in this sandbox

This sandbox cannot reach `generativelanguage.googleapis.com` — a direct
test during this project's build hung past a 15-second timeout using the
SDK's default gRPC transport, rather than failing cleanly with an error.
That's exactly why `GeminiLLMClient` configures `transport="rest"`
instead (see `services/llm_service.py`'s module docstring) — REST
respects request timeouts properly, which matters for real restricted
networks too, not just this sandbox.

Because of this, **no actual Gemini API call has been made or verified
by automated testing** during this phase. Every test in
`tests/test_llm_service.py`, `tests/test_grounding_validator.py`,
`tests/test_explanation_service.py`, and `tests/test_safety_regression.py`
uses `FakeLLMClient` (`tests/_fake_llm.py`) — a deterministic stand-in
implementing the same interface, with no network dependency. Run
`scripts/test_gemini_live.py` yourself, with a real `GEMINI_API_KEY`, to
verify actual Gemini behavior against this project's real prompt and
schema.
