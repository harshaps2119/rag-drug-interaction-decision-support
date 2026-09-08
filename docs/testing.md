# Testing — Project-Wide Summary (through Phase 8)

## How to run everything

**Backend:**
```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
pytest -v
```

**Frontend:**
```bash
cd frontend
npm install
npm test
npm run build
npm run lint
```

No live RxNorm, DailyMed, or Gemini API access is required for either
suite — everything is either a pure function test, a real local
resource (SQLite, ChromaDB) with no network dependency, or a
mocked/faked external call. The backend has one auto-skipping test for
the real embedding model (see below); the frontend suite has no
external dependencies of any kind.

## Current results

**Backend: 294 passed, 1 skipped** (`pytest -v` from `backend/`, ~15s
once dependencies are cached). The skip is
`tests/test_embeddings.py::test_real_model_loads_and_embeds_if_available`,
which requires network access to `huggingface.co` that this project's
build sandbox doesn't have — it's designed to run and pass when you run
it yourself with normal network access, not to be a permanent gap.

**Frontend: 54 passed, 0 failed** (`npm test` from `frontend/`, ~13s,
6 test files, no skips, no warnings). Production build (`npm run
build`) succeeds cleanly (46 modules, ~155KB JS / ~7KB CSS before
gzip). Static analysis (`npm run lint`, via `oxlint`) passes with 0
warnings and 0 errors.

## Testing approach by layer

| Layer | Approach |
|---|---|
| RxNorm/DailyMed services (Phase 2-3) | Mocked HTTP via `respx` |
| SQLite knowledge base (Phase 4) | Real in-memory SQLite, no mocking needed |
| Chunking (Phase 5) | Pure functions, no dependencies |
| Embeddings (Phase 5) | `DeterministicFakeEmbeddingModel` for plumbing; one auto-skipping real-model test |
| ChromaDB / RAG ingest (Phase 5) | Real local ChromaDB (tmp_path) + fake embeddings |
| Pair retrieval (Phase 6) | Real SQLite + real ChromaDB + fake embeddings, controlled fixtures |
| LLM / grounding validation (Phase 7) | `FakeLLMClient` — no real Gemini call |
| API layer (Phase 8) | Real FastAPI app (`TestClient`) with all dependencies overridden by fakes |
| Frontend (Phase 9) | React Testing Library with `services/api.js` mocked — no real backend or Gemini needed |

The guiding principle throughout: **use the real thing whenever it's
free and local** (SQLite, ChromaDB, pure functions), and **fake only
what genuinely requires network access or an API key** (RxNorm,
DailyMed, the embedding model download, Gemini) — never mock something
that doesn't need it, since that would test less than what's actually
possible to verify.

## Bugs found by actually running tests, not just writing them

This project's build process treats a test that hasn't been executed as
unverified, regardless of how carefully it was written. The following
were all caught this way, across every phase:

- **Phase 2:** a dependency version mismatch that silently broke HTTP
  mocking; a retry-vs-exception-translation ordering bug that disabled
  retries entirely.
- **Phase 4:** an incorrect test assumption about link-status reporting
  during idempotent re-ingestion (the test was wrong, not the code).
- **Phase 5:** ChromaDB rejects empty metadata dicts — harmless for
  production code (which never builds one), but required fixing several
  test fixtures.
- **Phase 7:** a test that forgot to set a required field, causing an
  unrelated validator check to fire first (fixed the test).
- **Phase 8, several in one phase:**
  - **In-memory SQLite doesn't share state across connections** without
    `StaticPool` — invisible in every earlier phase's tests (each used
    exactly one session per test) but broke immediately under FastAPI's
    per-request session pattern. Fixed in `app/db.py`, not worked around
    in tests.
  - **Monkeypatching a module attribute doesn't affect an
    already-`from`-imported name** in another module — the interaction
    router imported `check_pair` directly, so patching
    `interaction_service.check_pair` had no effect on the router's own
    reference. Fixed by having the router call through the module
    (`interaction_service.check_pair(...)`), which is also better
    practice generally.
  - **`TestClient`'s default `raise_server_exceptions=True`** re-raises
    exceptions for interactive debugging instead of returning the actual
    HTTP response a real deployment would give — made a working 500
    handler look broken in tests until the fixture explicitly opted out.
  - **A test rate-limit override recreated a fresh limiter on every
    call** (`lambda: InMemoryRateLimiter(...)`), so state never
    accumulated across requests and the limit never triggered — fixed by
    capturing one shared instance.
  - Four unused imports flagged by `pyflakes` (a real static-analysis
    pass, not just `py_compile`), including one pre-existing since
    Phase 2.
- **Phase 9 (frontend):**
  - A test pre-set `DrugSearch`'s input value via `useState` without
    ever focusing the field — the suggestion dropdown correctly stayed
    closed (by design: it only opens on focus/typing, not merely
    because a value is present), which looked like a component bug
    until the test itself was corrected to actually focus the input.
  - A test's own naive substring check flagged the frontend's OWN
    correctly-worded safety disclaimer ("this does not mean the
    medications are safe to combine") as an unsafe claim, because
    simple substring matching can't distinguish a negated warning from
    an affirmative one — the identical limitation already documented
    for the backend's forbidden-phrase validator in `docs/safety.md`.
    Fixed the test's regex to account for the negation, not the
    component (which was correct).
  - `oxlint` (static analysis) caught a genuinely unnecessary empty
    object literal spread fallback and one unused test import — both
    fixed.

None of these were hidden or silently patched around — each is fixed at
its actual source (application code where the bug was real, test code
where the test's assumption was wrong) and left documented here and in
the relevant module's docstring.

## Static analysis

```bash
cd backend
python3 -m pyflakes app/ tests/
cd ..
python3 -m pyflakes scripts/
```

Both pass clean (exit 0) as of Phase 8 — flags unused imports and
undefined names; run alongside `py_compile` (a syntax-only check) for a
cheap, genuinely useful pass beyond "does it parse."
