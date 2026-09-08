"""
db.py
======
SQLAlchemy engine and session management for the local SQLite knowledge base.

A BUG FIXED IN THIS PHASE
----------------------------
Phase 1's `.env.example` set SQLITE_DB_PATH=./backend/data/ddi.db, on the
implicit assumption that commands would always be run from the PROJECT
ROOT. But every script and README instruction in this project says
`cd backend` first — so a relative path like that would actually resolve
to `backend/backend/data/ddi.db`, which is wrong.

Fixed here: SQLITE_DB_PATH is now `data/ddi.db` (relative to backend/),
and `_resolve_db_path()` below anchors any relative path to the backend/
directory itself (using this file's own location), NOT the process's
current working directory. This means `python script.py` behaves
identically whether you run it from backend/ or from the project root —
removing an entire class of "works on my machine" bugs.

HOW TO USE
-----------
    from app.db import init_db, get_engine, get_session_factory

    init_db()  # creates tables if they don't exist yet — safe to call repeatedly
    Session = get_session_factory()
    session = Session()
    ...
    session.close()

For tests, pass an isolated in-memory engine instead:
    from app.db import make_engine, init_db, get_session_factory
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    Session = get_session_factory(engine)

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_db.py -v

This test file uses a real (in-memory) SQLite database — no mocking
needed, since SQLite is local and this sandbox has no network dependency
for this layer at all.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models.db_models import Base

# .../backend/app/db.py -> parent -> app/ -> parent -> backend/
BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _resolve_db_path(raw_path: str) -> Path:
    """Anchors a relative SQLITE_DB_PATH to the backend/ directory, not the cwd."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_engine(db_url: str | None = None) -> Engine:
    """
    Creates a new SQLAlchemy engine.
    - db_url=None (default): uses settings.SQLITE_DB_PATH, resolved relative
      to backend/.
    - db_url="sqlite:///:memory:": isolated in-memory database, for tests.

    BUG FIXED IN PHASE 8: for an in-memory SQLite URL, SQLAlchemy's default
    connection pool opens a NEW, completely separate (and empty) in-memory
    database for every new connection — meaning a second `Session()` from
    the same engine would not see tables/rows a first session created.
    This never surfaced in Phases 4-7's tests, which each used exactly one
    session for the lifetime of a test. It DOES surface under FastAPI's
    dependency-injection pattern (Phase 8), where every request gets a
    fresh `Session()` via `get_db_session()` — discovered by actually
    running a smoke test against the real app with TestClient, which
    failed with "no such table: drugs" despite the table having just been
    created and populated in a "first" session.

    Fixed by using `StaticPool` for in-memory URLs, which keeps and reuses
    the SAME underlying connection for every session from this engine —
    exactly what a shared in-memory test database needs. File-backed
    SQLite (the default/production path) is unaffected; StaticPool is
    only applied for `:memory:` URLs.
    """
    if db_url is None:
        db_path = _resolve_db_path(settings.SQLITE_DB_PATH)
        db_url = f"sqlite:///{db_path}"

    connect_args = {"check_same_thread": False}
    if ":memory:" in db_url:
        from sqlalchemy.pool import StaticPool

        return create_engine(db_url, connect_args=connect_args, poolclass=StaticPool)

    return create_engine(db_url, connect_args=connect_args)


_default_engine: Engine | None = None


def get_engine() -> Engine:
    """Returns the shared default engine (backed by the configured SQLITE_DB_PATH), creating it on first use."""
    global _default_engine
    if _default_engine is None:
        _default_engine = make_engine()
    return _default_engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker:
    """Returns a sessionmaker bound to the given engine, or the shared default engine if none is given."""
    return sessionmaker(bind=engine or get_engine(), autoflush=False, autocommit=False)


def init_db(engine: Engine | None = None) -> None:
    """Creates all tables if they don't already exist. Safe to call repeatedly (idempotent by design of CREATE TABLE IF NOT EXISTS)."""
    Base.metadata.create_all(bind=engine or get_engine())
