# Local Knowledge Base (SQLite)

Phase 4 adds a local SQLite database that stores everything Phase 2
(RxNorm) and Phase 3 (DailyMed) retrieve, so the system doesn't have to
re-hit those APIs on every request, and so every piece of evidence has a
permanent, traceable, queryable home.

## What this database is — and isn't

**It is an evidence/document store.** It records which drugs we've
normalized, which FDA labels we found for them, the specific label
sections we extracted with their original text, and a link connecting a
drug to that evidence.

**It is NOT a curated drug-interaction database.** It contains no
severity ratings, no "Drug A interacts with Drug B" relationships, and no
clinical judgment. Storing a label's Drug Interactions section text is
not the same as "we have identified an interaction between two specific
drugs" — that inference step doesn't exist yet (it will come later, in
Phase 6+, as an LLM reasoning over retrieved evidence, always with
citations back to rows in this database).

## Schema, in plain language

Five tables, forming a chain: **Drug -> DrugLabelLink -> Label -> LabelSectionRecord**,
plus **DrugSynonym** hanging off Drug.

### Drug
One row per unique RxCUI (Phase 2's output). Holds the canonical name,
term type (ingredient vs. brand, etc.), and which search term originally
found it.

### DrugSynonym
Brand names and other related names for a drug, so a search for
"Coumadin" finds the same drug stored under warfarin's RxCUI.

### Label
One row per unique DailyMed SPL, identified by `(setid, spl_version)` —
**not** just `setid` alone, because a label can be revised over time and
each revision is a legitimately different document version.

### LabelSectionRecord
One row per `(label, section)` pair — the actual evidence text, tagged
with its official LOINC section code. Rows are stored **even when a
section wasn't found** (`found=False`, empty text) — an explicit "we
checked and it wasn't there" is itself useful provenance, different from
"we never checked this label at all."

### DrugLabelLink
The join between a Drug and the Label DailyMed resolved for it. This is
where the `evidence_status` (`interaction_evidence_found` /
`label_found_no_interaction_section` / etc.) from Phase 3 is preserved —
**this is the only place a "relationship" is stored, and it's a
drug-to-evidence relationship, never a drug-to-drug one.**

## Why upserts, not plain inserts (idempotency)

Every table has a natural unique key enforced at the database level (not
just checked in application code):

| Table | Unique constraint |
|---|---|
| `drugs` | `rxcui` |
| `drug_synonyms` | `(drug_id, synonym, synonym_type)` |
| `labels` | `(setid, spl_version)` |
| `label_sections` | `(label_id, section_code)` |
| `drug_label_links` | `(drug_id, label_id)` |

`knowledge_base_service.ingest_drug()` always looks for an existing row
by its unique key first. If found and identical, it's reported
`"unchanged"` and nothing is written. If found but different (e.g. FDA
updated a label's wording), it's updated **in place** and reported
`"updated"`. Only genuinely new data is reported `"added"`. This is what
"running ingestion twice doesn't duplicate anything" actually means in
practice — verified in `tests/test_knowledge_base_service.py` by
ingesting the same drug three times and asserting row counts never grow
past one-per-entity.

One deliberate exception: `DrugLabelLink.retrieved_at` always refreshes
to the current time on every re-ingestion, even if nothing else changed —
this is a "last verified" timestamp, useful for knowing how fresh the
stored evidence is. It does **not** cause the link status to report
`"updated"` by itself; that only happens if `evidence_status` or the
candidate counts actually changed.

## Indexes (for later retrieval/filtering)

| Column | Why indexed |
|---|---|
| `drugs.rxcui` | Primary lookup key for "have we seen this drug" |
| `drugs.normalized_name` | Name-based search |
| `drug_synonyms.synonym` | Brand-name search |
| `labels.setid` | Lookup by DailyMed Set ID |
| `labels.manufacturer` | Future filtering by labeler |
| `label_sections.section_key` / `.section_code` | Filtering evidence by section type (e.g. "all Drug Interactions text") — this is exactly the filter Phase 6's retrieval will use |
| `label_sections.found` | Filtering to only meaningful evidence |
| `drug_label_links.evidence_status` | Filtering/reporting by evidence outcome |

## Provenance: tracing any stored text back to its source

Every `LabelSectionRecord` can be traced back to the exact FDA document
it came from via its parent `Label`:

```
LabelSectionRecord.text   (the actual evidence)
    -> section_code, section_name         (which part of the label)
    -> Label.setid, spl_version            (which document, which revision)
    -> Label.source_url                     (human-viewable DailyMed page)
    -> Label.api_url                        (the raw XML endpoint fetched)
    -> Label.manufacturer, published_date   (who published it, when)
    -> DrugLabelLink.evidence_status         (what kind of evidence this represents)
    -> Drug.rxcui, normalized_name            (which drug this evidence is for)
```

This chain is exactly what a later "View Evidence" UI feature and the
RAG/LLM layer's citations will walk to show the user where an answer
actually came from — see
`tests/test_knowledge_base_service.py::test_provenance_fields_preserved_end_to_end`
for a concrete example asserting every one of these fields survives
storage and retrieval intact.

## How Phase 4 connects the pipeline

```
normalize_drug_name()          [Phase 2 - RxNorm]
        |
        v
get_drug_label_info()          [Phase 3 - DailyMed]
        |
        v
ingest_drug()                   [Phase 4 - THIS PHASE]
        |
        v
Drug / Label / LabelSectionRecord / DrugLabelLink rows in SQLite
        |
        v
Phase 5 will read LabelSectionRecord.text rows, split them into chunks,
generate embeddings for each chunk, and store those embeddings in
ChromaDB - carrying along the same provenance metadata (section_code,
source_url, setid, drug rxcui, chunk id) as each embedding's metadata.
```

SQLite remains the durable, human-readable source of truth. ChromaDB
(Phase 5) will hold a derived, rebuildable *search index* over this same
data — if the vector store is ever corrupted or needs a schema change,
it can be regenerated from SQLite without re-hitting RxNorm or DailyMed.

## Path resolution bug fixed in this phase

Phase 1's `.env.example` set `SQLITE_DB_PATH=./backend/data/ddi.db`,
assuming commands always run from the project root. But every script in
this project is run via `cd backend && python ...`, which would have
resolved that path to the wrong nested location
(`backend/backend/data/ddi.db`). Fixed: the default is now
`data/ddi.db`, and `app/db.py` anchors any relative path to the
`backend/` directory itself (via `Path(__file__)`), not the process's
current working directory — so it works identically regardless of where
you run a script from.

## Known limitations

1. **Bounded candidate checking carries through.** Phase 3 checks at most
   3 candidate labels per drug; only the label actually used gets stored.
   A 4th+ candidate with better evidence, if one exists, is never seen.
2. **No full-text search yet.** `search_drugs_by_name()` uses SQL `LIKE`
   matching — fine for exact-ish name lookup, but not the semantic
   retrieval Phase 6 will need over section *content*. That's what
   ChromaDB (Phase 5) is for.
3. **Single local SQLite file.** Fine for development and an academic
   MVP; would need a real concurrent-access database (e.g. PostgreSQL)
   for multi-user production use.
4. **The seed set is not clinical data.** `app/data/seed_drugs.py`'s
   5-drug list exists purely to exercise the pipeline during development
   — see that file's docstring for the explicit disclaimer.
