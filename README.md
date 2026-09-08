# Drug–Drug Interaction (DDI) RAG Decision Support System

An evidence-grounded, citation-backed drug interaction checker built with a
hybrid **structured lookup + Retrieval-Augmented Generation (RAG)**
architecture. Built as an academic + portfolio project for Hospital/Health IT.

> ⚠️ **Not a medical device. Not for autonomous prescribing.**
> This is an educational/clinical-information prototype. It surfaces
> evidence from RxNorm and DailyMed (FDA structured product labels) and uses
> an LLM only to *explain retrieved evidence* — never to invent facts.
> Clinical judgment by a qualified professional is always required.

## Status

🚧 **Phase 9 of 15 complete: React frontend.**
See `docs/how-it-was-built.md` for the full build log, `docs/frontend.md`
for the frontend architecture, `frontend/README.md` for how to run it,
`docs/api.md` for the API reference, `docs/security.md` for the
hardening/privacy design, `docs/testing.md` for the project-wide testing
approach, and `docs/{rag,embeddings,retrieval,llm,safety}.md` for the
retrieval/LLM layers underneath it.

| Phase | Description | Status |
|---|---|---|
| 1 | Project setup | ✅ Done |
| 2 | RxNorm integration | ✅ Done |
| 3 | DailyMed integration | ✅ Done |
| 4 | Local knowledge base | ✅ Done |
| 5 | Embeddings + ChromaDB | ✅ Done |
| 6 | DDI retrieval & evidence assessment | ✅ Done |
| 7 | Gemini LLM + grounded explanation + safety validation | ✅ Done |
| 8 | Backend hardening, rate limiting, audit logging | ✅ Done |
| 9 | React frontend | ✅ Done |
| 10 | (merged into Phase 8 — see note below) | — |
| 11 | Multi-drug checking | ✅ Done (backend in Phase 8, UI in Phase 9) |
| 12 | Testing | ⏳ Ongoing every phase — see `docs/testing.md` |
| 13 | Evaluation | ⏳ Pending |
| 14 | Documentation | ⏳ Ongoing every phase |
| 15 | Final cleanup | ⏳ Pending |

**Notes on phase numbering:** this project's original 15-phase plan
named "FastAPI backend" and "Multiple-drug checking" as later, separate
phases. In practice, Phase 8's own instructions required a working
FastAPI backend with both a two-drug and a multi-drug endpoint to
implement rate limiting and audit logging meaningfully against — there
was no way to build request hardening without requests to harden. Both
are genuinely complete now, tracked honestly here rather than re-listed
as "pending" work that's already done. Phase 8 (as this conversation
numbers it) also already covers the grounding-validator work the
original plan's separate "Phase 8: Safety/validation layer" referred to.

## Architecture (high level)

```
User → React Frontend → FastAPI Backend → Drug Name Normalization (RxNorm)
     → RAG Retriever (ChromaDB, evidence from DailyMed labels)
     → LLM (Gemini) explains ONLY the retrieved evidence
     → Safety/Validation Layer → Final Answer + Citations
```

Full diagram: `docs/architecture.md` (added in Phase 14, stub now).

## Project structure

```
drug-interaction-rag/
├── backend/            FastAPI app, RAG pipeline, services, tests
├── frontend/            React + Vite UI
├── docs/                 Architecture, build log, report, viva prep
├── scripts/              Setup, ingestion, pipeline test scripts
├── .env.example        Copy to .env and fill in your own keys
└── .gitignore
```

## Quick start (once later phases add runnable code)

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp ../.env.example ../.env       # then edit .env with your GEMINI_API_KEY

# Initialize the local knowledge base (pure local SQLite, no network needed)
python ../scripts/setup_data.py

# Ingest the dev seed set (real network calls to RxNorm + DailyMed)
python ../scripts/ingest_labels.py --seed

# Build the vector index (real network call to download the embedding model)
python ../scripts/build_vector_index.py

# Verify the real Gemini API works with your key (optional but recommended)
python ../scripts/test_gemini_live.py

uvicorn app.main:app --reload --port 8000
```

Then, with the server running:
```bash
curl http://localhost:8000/api/health

curl -X POST http://localhost:8000/api/interaction/check \
  -H "Content-Type: application/json" \
  -d '{"drug_a": "warfarin", "drug_b": "ibuprofen"}'
```
Or visit http://localhost:8000/docs for interactive Swagger UI. Full
API reference: `docs/api.md`.

Run the test suite (offline — mocked HTTP + real in-memory SQLite/ChromaDB + fake LLM client, no API keys needed):
```bash
cd backend
pytest -v
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env    # only needed if your backend isn't at localhost:8000
npm run dev
```
Open the URL Vite prints (typically `http://localhost:5173`) in your
browser, with the backend (above) running alongside it. Full frontend
docs: `frontend/README.md` and `docs/frontend.md`.

Run the frontend test suite (mocked API, no backend needed):
```bash
cd frontend
npm test
npm run build   # production build
npm run lint     # static analysis
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Data sources

- **RxNorm** (NLM, public API, no key) — drug name normalization, RxCUI,
  brand/generic mapping. https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
- **DailyMed** (NLM/FDA, public API, no key) — Structured Product Labels:
  interactions, contraindications, warnings, pharmacology.
  https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/

Neither requires an API key. The **Gemini API key is the only credential
you need to supply** (free tier available at Google AI Studio).

See `docs/data-sources.md` for full endpoint documentation, LOINC section
codes used, and known coverage limitations of both sources.

## Safety principles this system follows

1. Never claims "no interaction exists" — only "no interaction identified
   in the searched sources," which is a meaningfully different (and honest)
   statement.
2. Every claim is traceable to a retrieved source chunk, shown to the user.
3. Severity is only shown if the underlying source actually states it —
   never inferred or invented.
4. If retrieval finds nothing relevant, the system says so rather than
   letting the LLM fill the gap from its own training data.

See `docs/limitations.md` (Phase 14) for the full list of known gaps.

## License

Add your preferred license (e.g., MIT) in `LICENSE` before publishing.
