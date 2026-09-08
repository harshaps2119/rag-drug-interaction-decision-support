# Data Sources

This document explains the two external data sources this system relies on,
exactly which API endpoints are used, and — critically for a clinical-safety
tool — their known limitations.

---

## RxNorm (Phase 2)

**What it is:** The National Library of Medicine's standardized nomenclature
for clinical drugs. Every drug concept gets a stable ID (RxCUI). We use it
to turn free-text drug names (generic, brand, or misspelled) into one
canonical identifier before doing anything else.

**Base URL:** `https://rxnav.nlm.nih.gov/REST` (no API key required)

**Endpoints used:**

| Endpoint | Purpose |
|---|---|
| `GET /rxcui.json?name={name}&search=2` | Exact/normalized name → RxCUI |
| `GET /approximateTerm.json?term={name}&maxEntries=5` | Fuzzy match for misspellings |
| `GET /rxcui/{rxcui}/property.json?propName=RxNorm Name` | Canonical display name |
| `GET /rxcui/{rxcui}/allrelated.json` | Related concepts (brand names, generic ingredient, synonyms) |

**How normalization decisions are made:** see `app/services/rxnorm_service.py`'s
module docstring for the full 3-tier lookup strategy (exact → approximate →
none) and why approximate matches below a 50/100 confidence score are
rejected rather than guessed.

---

## DailyMed (Phase 3)

**What it is:** The NLM/FDA's official repository of drug labeling —
Structured Product Labels (SPLs), which are the FDA-approved package
inserts, in a standardized HL7 XML format. This is where interaction,
contraindication, and warning text actually comes from.

**Base URL:** `https://dailymed.nlm.nih.gov/dailymed/services/v2` (no API
key required)

**Endpoints used:**

| Endpoint | Format | Purpose |
|---|---|---|
| `GET /spls.json?rxcui={rxcui}` | JSON | Search for labels by RxCUI (preferred — precise, from Phase 2) |
| `GET /spls.json?drug_name={name}` | JSON | Fallback search by free-text name |
| `GET /spls/{setid}.xml` | **XML only** | Full label content |

**Important technical detail:** DailyMed's *search* endpoint (`/spls.json`)
supports both XML and JSON. But the endpoint that returns the actual label
**content** (`/spls/{setid}`) supports **XML only** — there is no JSON
option for the full document. This is why `dailymed_service.py` parses HL7
SPL XML directly using Python's standard library `xml.etree.ElementTree`,
even though the search step uses JSON.

### How sections are identified

SPL documents use official FDA/LOINC codes to mark each section, e.g.:

| LOINC Code | Section |
|---|---|
| `34073-7` | Drug Interactions |
| `34074-5` | Drug and/or Laboratory Test Interactions |
| `34070-3` | Contraindications |
| `43685-7` | Warnings and Precautions |
| `34084-4` | Adverse Reactions |
| `34090-1` | Clinical Pharmacology |
| `34066-1` | Boxed Warning |

These codes are verified against FDA's official
["Section Headings (LOINC)"](https://www.fda.gov/industry/structured-product-labeling/section-headings-loinc)
reference, not assumed from memory. We match on the `<code code="...">`
element inside each `<section>`, which is far more reliable than matching
on section *titles* (titles vary in wording; codes don't).

### How "no interaction section" is distinguished from "no interaction"

This is the most important design decision in this phase, matching the
project's core safety requirement. Every label lookup returns one of four
`evidence_status` values:

1. **`interaction_evidence_found`** — a label was found and its Drug
   Interactions (or Drug/Lab Test Interactions) section has real content.
2. **`label_found_no_interaction_section`** — a label exists for this drug,
   but that section is absent or contains only placeholder text.
3. **`no_label_found`** — DailyMed has no SPL at all for this drug (by
   RxCUI or name).
4. **`error`** — the DailyMed API itself failed. This is *never* conflated
   with "no label found" — a broken API call says nothing about whether
   the drug or its interactions exist.

Downstream (Phase 6+), the RAG/LLM layer will use this status to phrase
its answer honestly: "no interaction identified in the searched sources"
(status 2 or 3) is a fundamentally different, weaker claim than "no
interaction exists" — and the system must never make the stronger claim.

### The placeholder "." problem

FDA SPL authoring guidance instructs label authors to enter a single "."
in a section if there's no content to add, rather than leaving it empty
(source: DailyMed/FDA SPL FAQ). Naively checking "does this section exist
in the XML" would therefore report false positives — a `.` is technically
present but conveys nothing. `dailymed_service._is_meaningful_text()`
explicitly filters out `.`, empty strings, `N/A`, `None`, `-`, etc. so
these are correctly treated as **absent**, not as evidence.

### Multiple manufacturers, multiple labels

DailyMed frequently has *several* SPLs for what a person would consider
"the same drug" — the original brand manufacturer's label, plus separate
labels filed by multiple generic repackagers. These can differ
significantly in completeness; a generic repackager's label sometimes has
sparse or missing sections while the original label is thorough.

Our approach: check up to **3 candidate labels** (`MAX_CANDIDATES_TO_CHECK`
in `dailymed_service.py`) per drug, in the order DailyMed's search returns
them, and use the first one that actually has interaction evidence. If
none of the checked candidates do, we report `label_found_no_interaction_section`
using the first candidate's other data, and note in `warnings` how many of
how many total candidates were checked.

**This is a documented trade-off, not a guarantee of completeness.** A
drug could theoretically have interaction evidence in a 4th+ candidate we
didn't check. This keeps lookups fast (bounded number of HTTP calls) at
the cost of possibly missing evidence in a long tail of duplicate labels.

### Source metadata preserved

Every `DrugLabelResult` retains, for full traceability back to the
original FDA document:

- `query_drug_name` / `query_rxcui` — what was actually searched for
- `setid` — DailyMed's Set ID (stable identifier for this label "family")
- `spl_version` — the specific version of the label used
- `title` — the label's official title
- `published_date` — from the search metadata
- `source_url` — human-viewable DailyMed page (`.../drugInfo.cfm?setid=...`)
- `api_url` — the raw XML API URL actually fetched
- per-section `section_code` (LOINC) and `section_name`

This lets a user (or a later UI "View Evidence" feature) trace any
displayed text all the way back to the exact FDA document and section it
came from.

---

## Known limitations of DailyMed/SPL interaction coverage

Documented honestly, per the project's core safety requirement:

1. **Not every label has interaction content.** SPL authors are not
   required to populate every section, especially for OTC/generic
   repackaged products. Absence of a Drug Interactions section is common
   and does not mean no interactions exist.
2. **No structured severity data.** SPL text is prose, not a coded
   severity rating. This system does not — and per the project
   requirements, must never — infer a severity level from this text via
   keyword or semantic matching. Any severity shown to the user in later
   phases must come from the source text explicitly stating it, if at all.
3. **Duplicate/inconsistent labels.** As above, multiple manufacturers can
   file separate SPLs for equivalent products with inconsistent
   completeness. We check a bounded number of candidates, not all of them.
4. **English-language, US-market only.** DailyMed only covers FDA-approved
   US drug labeling.
5. **Free text, not a structured interaction graph.** Unlike a licensed
   database (e.g. Lexicomp, Micromedex), DailyMed provides narrative label
   text. Extracting *which specific other drug* an interaction sentence
   refers to (versus a drug class, e.g. "NSAIDs") is a retrieval/NLP
   problem that Phase 6 (RAG retrieval) will address — this phase only
   retrieves the raw section text.
6. **No live verification of this phase's API assumptions from this
   sandbox.** RxNorm and DailyMed endpoint behavior in this document was
   verified against official NLM/FDA documentation web pages, but the
   sandbox this project was built in cannot reach either live API (see
   `scripts/test_dailymed_live.py` and `scripts/test_rxnorm_live.py`).
   Run those scripts yourself to confirm current live behavior matches
   what's documented here.
