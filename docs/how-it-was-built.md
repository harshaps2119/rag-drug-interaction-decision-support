# How This Was Built

This document explains the project from zero, in build order. It is
updated as each phase completes — currently covers Phases 1-4.

## Step 1 — Problem identification

Drug-drug interactions (DDIs) are a major, preventable source of patient
harm. A clinician or patient checking whether two medications interact
needs evidence-grounded information, not a guess — and needs to know the
difference between "no interaction was found in the sources we checked"
and "no interaction exists," because those are very different claims
with very different safety implications.

## Step 2 — System requirements

**Functional:** accept two (later, multiple) drug names; normalize them
to a canonical identifier; retrieve authoritative evidence about
interactions between them; explain that evidence without inventing
anything; show citations back to the original source.

**Non-functional:** must never fabricate medical information; must
distinguish "insufficient evidence" from "no interaction"; must be
traceable/auditable; should be inexpensive to run (free/public APIs where
possible); should be a real, editable codebase, not a black box.

## Step 3 — Technology selection

- **RxNorm & DailyMed** (Phase 2/3): free, official, public NLM/FDA APIs
  — no licensing cost, authoritative source, no API key required.
- **SQLite** (Phase 4): zero-configuration, file-based, perfect for a
  local dev/academic project; upgradeable to PostgreSQL later without
  changing the ORM layer (SQLAlchemy).
- **SQLAlchemy ORM**: type-safe schema definition, built-in migration
  path, works identically for the in-memory test database and the real
  file-based one.
- **ChromaDB** (Phase 5, upcoming): lightweight local vector database,
  no separate server process needed for an MVP.
- **FastAPI** (Phase 9, upcoming): async-friendly, automatic OpenAPI
  docs, strong Pydantic integration (which we're already using for every
  schema in this project).

## Step 4 — Data acquisition

Two-stage retrieval, in order:
1. **RxNorm** turns a free-text drug name into a stable RxCUI.
2. **DailyMed** is then queried BY that RxCUI (more precise than a
   name-only search) for the FDA-approved label, and the specific
   sections relevant to interaction checking are extracted from it.

Full endpoint documentation: `docs/data-sources.md`.

## Step 5 — Drug normalization

See `app/services/rxnorm_service.py`. Three-tier strategy: exact match,
then fuzzy/misspelling match (rejecting low-confidence guesses), then
"not found" — never silently guessing. Example: "worfarin" (typo) ->
approximate match -> warfarin (RxCUI 11289), flagged with a confidence
score so the caller/UI can show it as a suggestion, not a certainty.

## Step 6 — Knowledge-base creation

This is Phase 4. `app/services/knowledge_base_service.py`'s `ingest_drug()`
chains Step 5 (RxNorm) and Step 4's DailyMed retrieval into one
idempotent operation that stores everything in SQLite. See
`docs/database.md` for the full schema explanation.

## Step 7 — Chunking

`app/rag/chunking.py`. `LabelSectionRecord.text` rows from Phase 4 are
split into ~180-word overlapping windows (40-word overlap) — small
enough for the embedding model's input limit, large enough to keep a
complete thought together, with overlap so boundary-straddling sentences
aren't cut in half without context. A word-count sliding window was
chosen over sentence/paragraph splitting because Phase 3's section
extraction already flattens the original XML structure to plain text
(documented in that file), leaving no paragraph boundaries to split on,
and because medical text's abbreviations ("e.g.", "mg.") make naive
sentence splitting unreliable.

## Step 8 — Embeddings

`app/rag/embeddings.py`. Selected **all-MiniLM-L6-v2** — a
general-purpose, 384-dimension, ~90MB sentence embedding model that runs
on CPU with no GPU needed. This was a considered trade-off, not a
default: see `docs/embeddings.md` for the full comparison against
biomedical-domain alternatives (e.g. PubMedBERT-based models), which
would likely improve clinical-synonym matching at a meaningful cost in
size/complexity, and are documented as a future evaluation candidate
rather than dismissed. Because this sandbox's network blocks
huggingface.co, the real model could not be downloaded or tested here —
verified honestly with a direct failed-connection test, documented in
`docs/embeddings.md`, and the test suite reflects this split explicitly
(offline plumbing tests vs. one real-model test that cleanly skips).

## Step 9 — Vector database

`app/rag/vector_store.py`. **ChromaDB**, running as a local persistent
store (no separate server process). Embeddings are always computed by
this project's own code and passed to ChromaDB explicitly
(`embedding_function=None`), so ChromaDB itself never needs network
access — confirmed by running its tests fully offline in this sandbox.
Idempotency is built on ChromaDB's native `upsert()` combined with
deterministic chunk IDs derived from stable SQLite row IDs (see
`docs/rag.md` for the full explanation) — genuinely tested by ingesting
the same data three times and asserting the vector count never grows
past what a single ingestion produces.

## Step 10 — Retrieval

`app/services/pair_retrieval_service.py`. This is where "find evidence
similar to a query" (Phase 5) becomes "find evidence connecting Drug A
and Drug B specifically" — a hybrid strategy issuing several bounded,
deduplicated queries (pair-specific, drug-A-specific, drug-B-specific),
scoped via ChromaDB's `$in` filter to the relevant drug(s), then merging
and classifying results by `app/rag/entity_matching.py`'s simple
whole-word string matching against known drug names/synonyms and a
small illustrative drug-class vocabulary. The central design decision:
evidence is classified into `pair_specific_evidence` /
`class_level_evidence` / `drug_specific_evidence` /
`general_label_evidence` — never collapsed into a single "found
evidence" signal — specifically so two drugs that each have real
evidence, but whose labels never mention each other, are correctly
reported as `supporting_evidence_found`, never
`pair_specific_evidence_found`. See `docs/retrieval.md` for the full
reasoning and viva preparation.

## Step 11 — LLM integration

`app/services/llm_service.py`. Gemini (`google-generativeai`), used
strictly as an explanation layer over Phase 6's already-retrieved,
already-classified evidence — never as the source of what counts as
evidence. A modular `LLMClient` protocol (`.generate(prompt) -> str`)
means `GeminiLLMClient` is the only implementation touching Gemini
specifically; swapping providers later means writing one new small
class. A real, sandbox-discovered issue shaped the implementation:
the SDK's default gRPC transport did not respect its own configured
timeout when this sandbox's network couldn't reach
`generativelanguage.googleapis.com` — a direct test hung past a
15-second bash-level timeout instead of failing cleanly. Fixed by
configuring `transport="rest"`, which fails/succeeds within the
configured timeout properly — a fix that matters for real restrictive
networks too, not just this sandbox. See `docs/llm.md` for the full
prompt design, structured-output schema, and citation-handling
rationale.

## Step 12 — Safety layer

`app/services/grounding_validator.py`, running after every Gemini call
and before anything is shown to a user. Checks, independently of
anything the model claims about itself: every cited evidence id
actually exists; the claimed assessment never exceeds what Phase 6's
retrieval established; no "no interaction" language anywhere in the
free text; no fabricated URLs; and severity is verbatim-supported by
the cited evidence (the one check that's corrective — stripped, not
fatal — see `docs/safety.md` for why that asymmetry is deliberate). Any
fatal failure discards the entire response in favor of
`explanation_service.py`'s evidence-only fallback: full retrieved
evidence, fully cited, zero generated prose, with an explicit reason.
Every safety principle from earlier phases converges here: RxNorm's
low-confidence-match rejection, DailyMed's four-state evidence status,
Phase 4's refusal to store invented relationships, Phase 5's
"similarity is not clinical meaning" rule, and Phase 6's
never-upgrade-evidence classification are all mechanically re-enforced
at this final gate, not just assumed to have held all the way through.

## Step 13 — Backend

`app/main.py`, `app/api/`, `app/core/`. FastAPI, assembled from thin
route handlers (`app/api/{health,drugs,interaction}.py`) delegating to
services (`app/services/interaction_service.py` orchestrates Phases
2-7) — business logic never lives in route functions. Middleware order
matters and was verified, not assumed: `RequestIDMiddleware` must run
before `BodySizeLimitMiddleware`/`AuditMiddleware` so `request.state.request_id`
exists when they need it, proven by
`tests/test_middleware.py::test_request_id_available_to_body_size_middleware`.
A real, load-bearing bug was found and fixed while building this layer:
in-memory SQLite (`sqlite:///:memory:`) does not share state across
separate connections by default — invisible in every earlier phase's
tests (each used one session for a test's whole lifetime) but broken
immediately under FastAPI's per-request session pattern, where
`get_db_session()` opens a fresh session for every dependency
injection. Fixed properly in `app/db.py` with `StaticPool` for
in-memory URLs, not worked around in test code. See `docs/api.md` for
the full endpoint documentation.

## Step 14 — Frontend

React + Vite, `frontend/`. Small, single-purpose components composed in
one page (`pages/Home.jsx`) — no router needed for a single-page
prototype, no global state library needed since all state genuinely
belongs to one form or one list. `services/api.js` is the sole place
that calls `fetch`, normalizing every possible failure (network error,
non-2xx response, malformed JSON) into one `ApiError` type so every
component has exactly one error shape to handle. Full architecture,
component responsibilities, and safety-UX reasoning: `docs/frontend.md`.

**What was genuinely verified, and how, stated precisely:**

1. **54 frontend unit/integration tests, all executed, all passing**,
   using React Testing Library with `services/api.js` mocked — proves
   component behavior, composition, accessibility wiring (label/input
   association, ARIA roles), and safety-UX rules (e.g. "insufficient
   evidence" never renders alongside an affirmative "safe to combine"
   claim) against real rendered output.
2. **A real, live integration check beyond unit tests**: the actual
   FastAPI server was started in this build sandbox (real SQLite, real
   ChromaDB, real drug data seeded), and hit with `curl` requests
   shaped exactly like the frontend's `api.js` sends them. This
   confirmed the real JSON contract — and produced an honest, useful
   finding: the real `SentenceTransformerEmbeddingModel` was attempted
   (this is a genuine server process, not a `TestClient` with
   dependency overrides) and failed to download (no network access to
   huggingface.co, exactly as documented since Phase 5) — and the full
   safety architecture built across Phases 6-8 worked correctly under
   this REAL failure: the request returned a clean HTTP 200 with
   `evidence_status: "retrieval_error"`, `mode: "evidence_only"`, and a
   plain fallback reason — never a crash, never a stack trace, never a
   silently wrong answer. This is a stronger proof of the safety
   fallback path than any mocked test could offer, because the failure
   was real, not simulated.
3. **A real, discovered sandbox constraint, documented rather than
   hidden**: background processes (e.g. `uvicorn &`) do not persist
   between separate tool invocations in this build environment — each
   is a fresh process namespace. Discovered by watching a backgrounded
   server disappear between calls, worked around by running the server
   startup and every curl check within one single invocation.
4. **What was NOT done, stated plainly**: no real browser was launched.
   Visual rendering, mouse/keyboard interaction against the live app,
   and the two dev servers (`npm run dev` + `uvicorn --reload`) running
   side-by-side have not been observed by Claude. See the frontend's
   own `README.md` for the exact commands to verify this yourself.

## Step 15 — Testing

Every phase so far has real, executed tests, not just written-and-assumed
code:

| Phase | Test file | Tests | Approach |
|---|---|---|---|
| 2 | `test_rxnorm_service.py` | 10 | Mocked HTTP (respx) |
| 3 | `test_dailymed_service.py` | 16 | Mocked HTTP (respx) |
| 4 | `test_db.py` | 10 | Real in-memory SQLite, no mocking needed |
| 4 | `test_knowledge_base_service.py` | 15 | Real in-memory SQLite + mocked upstream calls |
| 5 | `test_chunking.py` | 9 | Pure functions, no dependencies |
| 5 | `test_embeddings.py` | 7 (6 real + 1 auto-skip) | Offline fake model + one real-model test that skips cleanly without network |
| 5 | `test_vector_store.py` | 18 | Real local ChromaDB, no network needed |
| 5 | `test_rag_ingest.py` | 13 | Real in-memory SQLite + real ChromaDB + fake embeddings |
| 5 | `test_retrieval_service.py` | 10 | Real ChromaDB + fake embeddings |
| 6 | `test_entity_matching.py` | 15 | Pure functions, no dependencies |
| 6 | `test_pair_retrieval_service.py` | 25 | Real in-memory SQLite + real ChromaDB + fake embeddings, controlled fixtures |
| 7 | `test_llm_service.py` | 19 | FakeLLMClient — prompt content, parsing, exception translation |
| 7 | `test_grounding_validator.py` | 17 | Pure function tests — all 5 validator checks |
| 7 | `test_explanation_service.py` | 17 | FakeLLMClient — orchestration, all fallback modes |
| 7 | `test_safety_regression.py` | 10 | The 6 spec-mandated named safety regression tests |
| 8 | `test_rate_limiter.py` | 9 | Pure logic, injectable clock |
| 8 | `test_audit_log.py` | 7 | JSON formatting + signature-based privacy checks |
| 8 | `test_middleware.py` | 7 | Real minimal FastAPI app — request ID, body size, ordering |
| 8 | `test_interaction_service.py` | 6 | Real SQLite + ChromaDB + fake embeddings, mocked ingestion |
| 8 | `test_api_validation.py` | 31 | Pydantic model tests + through-the-endpoint validation |
| 8 | `test_health_endpoint.py` | 5 | Real app via TestClient |
| 8 | `test_error_handling.py` | 5 | Real app — secret leakage prevention |
| 8 | `test_interaction_router.py` | 14 | Real app, full endpoint integration, rate limiting, CORS |
| 9 | `services/api.test.js` | 12 | Vitest — mocked `fetch`, all error paths |
| 9 | `components/DrugSearch.test.jsx` | 7 | React Testing Library — autocomplete, ARIA, no raw IDs shown |
| 9 | `components/EvidenceCard.test.jsx` | 6 | Expand/collapse, source link fidelity |
| 9 | `components/InteractionResult.test.jsx` | 11 | All evidence_status/mode combinations, severity fallback |
| 9 | `components/MultiDrugChecker.test.jsx` | 9 | Add/remove, duplicate prevention, submit |
| 9 | `pages/Home.test.jsx` | 9 | Full page integration — form validation, all states |

**Total as of Phase 8 (backend): 295 tests — 294 passing, 1 cleanly skipped.**
**Total as of Phase 9 (frontend): 54 tests, all passing** (`npm test`
from `frontend/`, ~13s, 6 files). Backend and frontend test suites are
run and reported separately — they exercise different runtimes
(pytest/Python vs. Vitest/JS) and are never combined into one number.
The one backend skip is the real-embedding-model test (Phase 5); the
real-Gemini path (Phase 7) and Phase 8/9's own external-dependency gaps
are covered by live scripts you run yourself, not auto-skipping tests,
per each phase's own documentation. Full per-layer testing approach and
the complete, unabridged list of bugs found by actually running tests:
`docs/testing.md`.

Phase 8 alone surfaced four separate real bugs by actually running its
tests (not just writing them) — most notably the in-memory-SQLite
`StaticPool` fix (a genuine application bug, fixed in `app/db.py`) and a
subtle Python monkeypatching gotcha in the interaction router (fixed by
having it call through the module rather than importing the function
directly, which is also better practice) — see `docs/testing.md` for
the complete list including two test-only fixes (TestClient's exception
re-raising default, and a rate-limit test that recreated a fresh limiter
every call).

Phase 9 surfaced its own real findings: a test that pre-set an input's
value via `useState` without focusing it, silently failing to trigger
`DrugSearch`'s open-suggestions behavior (which correctly requires focus,
by design) — fixed the test, not the component; and a test's own naive
substring check flagging the frontend's OWN correctly-worded safety
disclaimer ("this does not mean the medications are safe to combine")
as if it were an unsafe claim, because a bare substring match can't
distinguish a negated warning from an affirmative one — the exact same
limitation documented for the backend's own forbidden-phrase validator
in Phase 7's `docs/safety.md`. Both are detailed in `docs/testing.md`.

## Step 16 — Evaluation

*(Phase 13, upcoming.)*

## Step 17 — Deployment

*(Phase 15, upcoming — for now, everything runs locally; see each
phase's "how to run" instructions in this README and the relevant
`docs/*.md` files.)*
