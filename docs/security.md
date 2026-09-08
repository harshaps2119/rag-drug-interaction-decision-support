# Security & Hardening — Phase 8

This document covers application-level security added in Phase 8, the
production trade-offs made honestly rather than hidden, and prepared
viva answers.

## Scope decision, stated up front

This project is an **academic prototype**, not a hospital-grade
production system. Phase 8 adds real, working hardening appropriate to
that scope — input validation, rate limiting, audit logging, consistent
error handling, CORS configuration, a real health check — but does
**not** add enterprise features that would only be theater at this
stage: no Redis-backed distributed rate limiting, no authentication/
authorization system, no encryption-at-rest configuration, no WAF, no
intrusion detection. Where something is only suitable for local
development, this document says so explicitly rather than dressing it
up as production-ready.

## Privacy by design: this project collects no patient information

This is worth stating plainly: **the only user input anywhere in this
system is drug names.** No patient names, patient IDs, phone numbers,
addresses, or medical record numbers are ever collected, stored, or
have any code path to reach a log file. This isn't a policy promise
layered on top of a system that could technically log such things — the
audit logging functions (`app/core/audit_log.py`) are written with a
FIXED, explicit set of named parameters (request ID, endpoint, RxCUI,
evidence status, latency, error category) and no `**kwargs`
passthrough anywhere. There is no code path capable of accidentally
logging free text a caller sent, an API key, or anything resembling
patient data — verified directly by
`tests/test_audit_log.py::test_audit_functions_have_no_kwargs_passthrough`
and `::test_audit_functions_do_not_expose_forbidden_parameter_names`,
which inspect the actual function signatures, not just their current
call sites.

Drug names themselves are logged as their resolved **RxCUI** (a
structured identifier), not the free-text string the caller typed — one
further step of sanitization, and consistent with treating any raw user
input as untrusted for logging purposes.

## Input validation

See `docs/api.md` for the exact rules. The design principle: be
permissive about what makes a **valid drug name** (real names contain
hyphens, slashes, parentheses, apostrophes — rejecting those would break
legitimate use), while being strict about excluding characters with no
legitimate reason to appear in a drug name and every reason to be
treated as suspicious in a field that flows into a database query and an
LLM prompt (`<`, `>`, `;`, `` ` ``, `{`, `}`).

## Rate limiting

**Implementation:** in-memory, per-process, sliding-window
(`app/core/rate_limiter.py`), keyed by client IP, applied only to the
two interaction endpoints (never to `/api/health`, which needs to be
pollable frequently).

**Why in-memory, not Redis:** this phase's own scope instructions say to
prefer a simple local/development-compatible implementation unless
there's a strong reason otherwise, and to explain the trade-off rather
than hide it. Here it is, plainly:

> This limiter's state lives in ONE Python process's memory. Running
> multiple uvicorn workers or multiple container instances behind a load
> balancer means each one enforces the limit INDEPENDENTLY — a client
> could get `RATE_LIMIT_MAX_REQUESTS` requests **per worker/instance**,
> not in total, and a process restart resets everyone's count to zero.
> A real multi-instance production deployment needs a shared backing
> store (Redis, typically) so the limit is enforced across the whole
> fleet atomically. That is deliberately not built here — adding it
> without an actual multi-instance deployment to justify it would be
> exactly the "fake enterprise feature" this phase's scope explicitly
> warns against.

Configurable via `.env`: `RATE_LIMIT_ENABLED`, `RATE_LIMIT_MAX_REQUESTS`,
`RATE_LIMIT_WINDOW_SECONDS` — defaults (30 requests/60s) are generous
enough that normal local development and manual API testing aren't
hampered, per the phase's explicit requirement.

## Request IDs

Every request gets a correlation ID (`app/core/request_id.py`) — reused
from an incoming `X-Request-ID` header if the caller supplies one,
otherwise a fresh UUID4. It's echoed in the `X-Request-ID` response
header, included in every error envelope's `request_id` field, and
attached to every audit log line for that request. This is the single
key that lets a specific user-reported issue be traced through logs
without guessing based on timestamps alone.

## Audit logging

Structured, one-JSON-line-per-event, via `app/core/audit_log.py`. Two
kinds of events:

1. **Generic, every request** (`AuditMiddleware`): method, path, status
   code, latency, request ID, client host.
2. **Domain-specific, interaction checks only**: RxCUIs (not raw drug
   names), evidence status, LLM mode (`llm_grounded` /
   `evidence_only`), latency, error category.

See "Privacy by design" above for what's structurally excluded.

## API error handling

Every error response — regardless of source — has the exact same shape:
`{"error": {"code": "...", "message": "...", "request_id": "..."}}`.
`message` is always a short, fixed, safe string; the generic 500 handler
(`app/core/errors.py::unhandled_exception_handler`) builds its response
body from a hard-coded string, NEVER by interpolating the actual
exception's text — so there's no risk of an unexpected exception
accidentally leaking a file path, a partial query, or anything else
sensitive. The full exception (with traceback) is logged server-side via
`logger.exception(...)`, visible only in server logs. Verified directly
by `tests/test_error_handling.py`, which deliberately raises an
exception containing a fake secret and confirms it never appears
anywhere in the HTTP response.

## CORS

Configured via `settings.FRONTEND_ORIGIN` (from `.env`), allowing only
that one origin, `GET`/`POST` methods, and `allow_credentials=False` (a
safer default — this API uses no cookies/session auth to protect).
Appropriate for a single local frontend during development; a real
multi-origin or public deployment would need this reviewed and likely
tightened (e.g. an explicit origin allowlist per environment, HTTPS-only
origins).

## Debug mode

`settings.DEBUG` (default `False`) is a dev-only convenience flag,
currently only affecting FastAPI's own `debug=` constructor argument
(which controls whether Starlette shows its own debug traceback pages
for framework-level errors). **Never enable this in a real deployment**
— it can reveal internal details FastAPI's default debug page shows.

## What would need to change before any real hospital deployment

Stated honestly, not glossed over:

1. **Authentication & authorization.** This API currently has none —
   anyone who can reach it can use it. A real deployment needs at
   minimum API-key or OAuth-based auth, and likely role-based access
   control if integrated with clinical staff workflows.
2. **Distributed rate limiting** (Redis-backed) if running more than one
   process/instance.
3. **TLS termination** in front of the API (not built or assumed here).
4. **Secrets management** — `.env` files are fine for local development;
   a real deployment needs a proper secrets manager (e.g. cloud
   provider's secret store), not a file on disk.
5. **Structured log shipping** to a real log aggregation/SIEM system,
   with retention policy decisions made by the deploying organization.
6. **A production-grade database** (Phase 4's `docs/database.md` already
   flags this) — SQLite is fine for a prototype, not for concurrent
   multi-user production load.
7. **Clinical validation and regulatory review** — entirely outside this
   project's scope, and the most important gap of all: nothing here has
   been clinically validated, and it must not be treated as a medical
   device or relied upon for real clinical decisions in its current
   form.
8. **CORS/host allowlisting** tightened to the real deployed frontend
   origin(s), not a single dev URL.

## Viva preparation

**"Why do we need rate limiting?"** To prevent a single client
(accidental script loop, or intentional abuse) from monopolizing shared
resources — especially the Gemini-backed endpoint, which has a real
per-call cost and a provider-side rate limit of its own that a local
loop could exhaust for everyone.

**"Why do we need audit logs?"** Operational visibility: which
endpoints are used, how often, how long they take, and what kind of
result (evidence-only vs. LLM-grounded) they produced — essential for
debugging, capacity planning, and noticing abuse patterns, all without
needing to know anything about who's asking.

**"What information should not be logged?"** Patient names, patient
IDs, phone numbers, addresses, medical record numbers, API keys, raw
secrets — and, as a general principle, any unnecessary free text. This
project also never collects any of the patient-identifying items in the
first place, only drug names, so the practical rule here narrows to: log
structured identifiers (RxCUI), never the raw string a caller typed, and
never a secret.

**"What is a request ID?"** A correlation identifier attached to one
HTTP request, propagated through logs, audit records, and error
responses, so a single request's full trail can be found without
ambiguity — see "Request IDs" above.

**"How do you protect API keys?"** Read from `.env` (never committed,
covered by `.gitignore` since Phase 1) into `settings.GEMINI_API_KEY`,
never logged (see audit logging design), and never included in any error
response body (see error handling design, and the executed test proving
a fake key injected into a broken response never appears in the client-
facing output).

**"What is CORS?"** Cross-Origin Resource Sharing — a browser-enforced
mechanism controlling which web origins are allowed to make requests to
this API from client-side JavaScript. Configured here to allow only the
one configured frontend origin.

**"How does your backend handle API failures?"** Every external
dependency (RxNorm, DailyMed, the embedding model, Gemini) has its own
typed exceptions from earlier phases; at the API layer, any failure that
reaches `interaction_service.py` either resolves to a specific,
documented `evidence_status` (e.g. `drug_not_found`, `retrieval_error`)
or, for a genuinely unexpected error, a safe generic 500 — never a raw
exception or a silently wrong answer.

**"Why shouldn't health checks call Gemini?"** A health check needs to
be fast, cheap, and reflect whether THIS service can serve requests —
not whether a third-party API happens to be having a slow five minutes.
Calling Gemini on every health check would also burn real API cost on
every poll, make health status dependent on an external SLA this project
doesn't control, and risk an orchestrator restarting a perfectly healthy
service because of someone else's outage.

**"How would you secure this in a real hospital environment?"** See "What
would need to change before any real hospital deployment" above —
authentication/authorization, TLS, a real secrets manager, distributed
rate limiting, log shipping to a compliant SIEM, a production database,
and — the actual hard requirement — formal clinical validation and
regulatory review before any of this touches real patient care.

**"What changes would be required before production deployment?"** The
same list — this project remains a prototype until all of it is
addressed, not just the technical items.
