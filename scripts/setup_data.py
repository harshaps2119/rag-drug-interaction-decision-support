"""
scripts/setup_data.py
========================
Initializes the local SQLite knowledge base — creates all tables if they
don't already exist yet. Safe to run repeatedly (idempotent: CREATE TABLE
IF NOT EXISTS under the hood).

This script does NOT make any network calls — it's pure local SQLite
setup, so it runs identically here and on your machine.

HOW TO RUN
-----------
    cd backend
    python ../scripts/setup_data.py

EXPECTED OUTPUT
-----------------
    Database initialized at: sqlite:////.../backend/data/ddi.db
    (Configured SQLITE_DB_PATH: data/ddi.db)
    Tables: drugs, drug_synonyms, labels, label_sections, drug_label_links
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.db import get_engine, init_db  # noqa: E402
from app.models.db_models import Base  # noqa: E402


def main() -> None:
    engine = get_engine()
    init_db(engine)
    table_names = sorted(Base.metadata.tables.keys())
    print(f"Database initialized at: {engine.url}")
    print(f"(Configured SQLITE_DB_PATH: {settings.SQLITE_DB_PATH})")
    print(f"Tables: {', '.join(table_names)}")


if __name__ == "__main__":
    main()
