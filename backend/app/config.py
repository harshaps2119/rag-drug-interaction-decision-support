"""
config.py
=========
Central configuration for the whole backend.

WHY THIS FILE EXISTS
---------------------
Every service (RxNorm client, DailyMed client, LLM client, vector store)
needs settings like API URLs, model names, and file paths. Instead of each
file reading os.environ directly (which is error-prone and hard to test),
we define ONE typed settings object here using pydantic-settings.

Supported LLM_PROVIDER values:
  xai        — xAI Grok via OpenAI-compatible API (default)
  gemini     — Google Gemini via google-generativeai SDK
  openrouter — Any OpenRouter-hosted model via OpenAI-compatible API
  none       — Disable LLM; always use evidence-only mode

pydantic-settings automatically:
  1. Reads variables from the `.env` file at project root.
  2. Validates their types (e.g. BACKEND_PORT must be an int).
  3. Gives you autocomplete in your editor (settings.GEMINI_API_KEY etc.)
  4. Fails loudly and clearly if something required is missing/misspelled,
     instead of silently returning None deep inside some unrelated function.

HOW TO USE
----------
    from app.config import settings
    print(settings.RXNORM_BASE_URL)

TEST IT
-------
    cd backend
    python -c "from app.config import settings; print(settings.RXNORM_BASE_URL)"

Expected output:
    https://rxnav.nlm.nih.gov/REST
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- LLM ----
    # xAI is the default explanation provider. Gemini and OpenRouter are
    # optional providers. 'none' disables LLM explanations (evidence-only mode).
    LLM_PROVIDER: Literal["xai", "gemini", "openrouter", "none"] = "xai"
    XAI_API_KEY: str = ""
    XAI_MODEL: str = "grok-4.6"
    XAI_BASE_URL: str = "https://api.x.ai/v1"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # OpenRouter — uses OpenAI-compatible API; supports hundreds of models.
    # See https://openrouter.ai/models for available model identifiers.
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # ---- RxNorm ----
    RXNORM_BASE_URL: str = "https://rxnav.nlm.nih.gov/REST"

    # ---- DailyMed ----
    DAILYMED_BASE_URL: str = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

    # ---- Local data stores ----
    # Relative paths are resolved against the backend/ directory by
    # app/db.py, regardless of the process's current working directory.
    SQLITE_DB_PATH: str = "data/ddi.db"
    CHROMA_PERSIST_DIR: str = "data/chroma_store"

    # ---- Embeddings ----
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # ---- RAG chunking & retrieval (Phase 5) ----
    RAG_CHUNK_SIZE_WORDS: int = 180
    RAG_CHUNK_OVERLAP_WORDS: int = 40
    RAG_DEFAULT_TOP_K: int = 5

    # ---- LLM grounded explanation (Phase 7) ----
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_EVIDENCE_ITEMS: int = 8

    # ---- Backend hardening (Phase 8) ----
    DEBUG: bool = False  # dev-only convenience; never enable in production (see docs/security.md)

    # Rate limiting -- in-memory, per-process (see docs/security.md for the
    # production trade-off). Generous defaults so local development isn't
    # hampered; tighten for any shared/public deployment.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_MAX_REQUESTS: int = 30
    RATE_LIMIT_WINDOW_SECONDS: float = 60.0

    # Request validation
    MAX_DRUG_NAME_LENGTH: int = 200
    MAX_DRUGS_PER_MULTI_REQUEST: int = 10
    MAX_REQUEST_BODY_BYTES: int = 20_000  # 20KB -- generous for a JSON body of drug names

    # ---- Server ----
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8001
    FRONTEND_ORIGIN: str = "http://localhost:5173"

    # Look for a ".env" file two levels up from this file's typical run
    # location (i.e. project root), but also fall back to a local .env
    # if the app is run from inside backend/.
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# A single shared instance imported everywhere else in the app.
settings = Settings()
