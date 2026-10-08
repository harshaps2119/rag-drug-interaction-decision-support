# API Documentation — Phase 8

## Request flow

```
HTTP request
    |
    v
RequestIDMiddleware        -- assigns/reuses X-Request-ID
    |
    v
AuditMiddleware              -- times the request, logs one structured line on completion
    |
    v
BodySizeLimitMiddleware       -- rejects oversized bodies before they're parsed
    |
    v
CORSMiddleware                  -- enforces configured allowed origin
    |
    v
Route handler (thin -- delegates to a service)
    |
    v
rate_limit_dependency (interaction endpoints only)
    |
    v
Pydantic request validation (schemas/api.py)
    |
    v
services/interaction_service.py
    -> ensure_drug_available()   [Phases 2-4, on demand]
    -> ChromaDB indexing          [Phase 5, on demand]
    -> PairRetrievalEngine         [Phase 6]
    -> generate_explanation()       [Phase 7: selected LLM + grounding validation]
    |
    v
Structured JSON response (ExplanationResponse / MultiDrugCheckResponse)
```

Every step from here down (Phases 2-7) is unchanged from earlier phases —
Phase 8 only adds the HTTP layer, hardening, and the orchestration glue
in `interaction_service.py` that ties them together for a live request.

## Endpoints

### `GET /api/health`

Reports whether THIS service's own local dependencies work — SQLite,
ChromaDB, and whether the selected LLM API key is configured. **Never calls
RxNorm, DailyMed, or an LLM provider itself** (see `docs/security.md` for why).

```json
{
  "status": "ok",
  "checks": {
    "sqlite": {"status": "ok", "detail": null},
    "chromadb": {"status": "ok", "detail": null},
    "llm_configured": {"status": "ok", "detail": null}
  },
  "request_id": "..."
}
```

### `GET /api/drugs/search?q=<name>`

Read-only search against the local SQLite knowledge base (Phase 4). No
external calls. Supports a future frontend's drug-name autocomplete.

### `POST /api/interaction/check`

The core two-drug endpoint. Rate-limited (see below).

**Request:**
```json
{"drug_a": "warfarin", "drug_b": "ibuprofen"}
```

**Response:** an `ExplanationResponse` (same schema as Phase 7) — either
`mode: "llm_grounded"` with a provider-generated, validated explanation, or
`mode: "evidence_only"` with the raw retrieved evidence and a
`fallback_reason`, per every failure mode documented in `docs/safety.md`.

### `POST /api/interaction/check-multiple`

**Request:**
```json
{"drugs": ["warfarin", "ibuprofen", "aspirin"]}
```

Generates all C(N,2) unique pairs (Phase 6's `generate_unique_pairs()`),
reusing Phase 6's per-drug query cache across pairs that share a drug.

**Response:**
```json
{
  "pairs": [
    {"drug_a": "warfarin", "drug_b": "ibuprofen", "result": { ...ExplanationResponse... }},
    {"drug_a": "warfarin", "drug_b": "aspirin", "result": { ... }},
    {"drug_a": "ibuprofen", "drug_b": "aspirin", "result": { ... }}
  ],
  "total_pairs": 3,
  "request_id": "..."
}
```

## Input validation

Drug names must be non-empty, at most `MAX_DRUG_NAME_LENGTH` (default
200) characters, and match a pattern allowing letters, digits, spaces,
and the punctuation real drug names legitimately use — hyphens,
periods, commas, parentheses, slashes, apostrophes, percent signs.
Examples correctly **allowed**: "Co-trimoxazole", "Vitamin B-12",
"Humalog 75/25", "St. John's Wort". Examples correctly **rejected**:
anything containing `<`, `>`, `;`, `` ` ``, `{`, `}` — characters with
no legitimate place in a drug name.

The multi-drug endpoint additionally requires 2-`MAX_DRUGS_PER_MULTI_REQUEST`
(default 10) drugs, with duplicate names (case-insensitive) rejected
outright rather than silently deduplicated. Full rules and rationale:
`app/schemas/api.py`'s module docstring.

Request bodies over `MAX_REQUEST_BODY_BYTES` (default 20KB) are rejected
at the middleware level, before parsing — this API has no legitimate
reason to receive a large JSON body for a couple of drug names.

## Error responses

Every error, from every source, has this exact shape:

```json
{"error": {"code": "...", "message": "...", "request_id": "..."}}
```

| Code | Status | When |
|---|---|---|
| `validation_error` | 422 | Malformed JSON, missing field, invalid drug name |
| `rate_limit_exceeded` | 429 | Too many requests from one client in the window |
| `request_too_large` | 413 | Request body exceeds the configured limit |
| `internal_error` | 500 | Any unexpected server-side error |

`message` is always a short, safe string. Stack traces, internal file
paths, and raw exception text are logged server-side only, never
returned to the client — see `docs/security.md`.

## Rate limiting

Applied to both interaction endpoints (not `/api/health` or
`/api/drugs/search`), configurable via `.env`:
`RATE_LIMIT_ENABLED`, `RATE_LIMIT_MAX_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`
(defaults: enabled, 30 requests per 60 seconds, per client IP). Full
implementation and production trade-off: `docs/security.md`.

## The clinical safety rule, enforced at this layer too

`interaction_assessment` in any response can only ever be one of:
`pair_specific_evidence_found`, `supporting_evidence_found`,
`insufficient_evidence`, `drug_not_found`, `invalid_input`,
`retrieval_error` — inherited
directly from Phases 6-7 and never weakened here. "No interaction
exists" is not a value this API can return under any circumstance,
including a Gemini outage or a retrieval failure — verified directly by
`tests/test_interaction_router.py::test_no_scenario_ever_returns_no_interaction_exists_phrase`.
