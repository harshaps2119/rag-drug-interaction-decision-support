# Pair Retrieval & Evidence Assessment — Phase 6

## The one-sentence explanation for your viva

> **Phase 5 taught the computer how to find similar evidence. Phase 6
> teaches it what evidence to look for when the question is specifically
> about Drug A interacting with Drug B.**

## Why finding two drugs' evidence separately is not enough

Suppose a user asks about Warfarin and Metformin. Phase 5 alone can
easily retrieve Warfarin's Drug Interactions text AND Metformin's Drug
Interactions text — both drugs are well-documented, both have real
label evidence. But **neither label mentions the other drug.** Reporting
"evidence found" for this pair, without checking whether that evidence
actually connects the two drugs, would be actively misleading — it would
look exactly the same, output-wise, as a real, connected interaction.

This is the central problem Phase 6 solves: **retrieving relevant
information about each drug individually is not the same as finding
evidence that the two actually interact.** Phase 6 never takes the
shortcut of assuming a connection just because both sides have content.
This is proven directly by an executed test:
`test_retrieve_pair_does_not_conclude_interaction_from_two_unrelated_labels`
in `tests/test_pair_retrieval_service.py` — two real (fixture) drug
labels, each with genuine interaction text, that never mention each
other, and the system correctly reports `supporting_evidence_found`,
never `pair_specific_evidence_found`.

## The retrieval strategy: hybrid, multi-query

A single query can't reliably find pair-relevant evidence, so
`generate_pair_queries()` builds several, of three kinds:

| Query type | Example | Scoped to |
|---|---|---|
| `pair_specific` | "warfarin and ibuprofen interaction" | both drugs' rxcuis (`$in` filter) |
| `drug_a_specific` | "warfarin drug interactions" | drug A's rxcui only |
| `drug_b_specific` | "ibuprofen drug interactions" | drug B's rxcui only |

Each drug's known brand names/synonyms (from Phase 2/4) are also used —
up to 2 name variants per drug, to keep the query set bounded rather
than growing combinatorially with every synonym on file. All queries run
through Phase 5's `retrieve()`, and results are merged and deduplicated
by `chunk_id` (a chunk retrieved by more than one query keeps its best
distance and records every query that found it).

## Evidence classification: the actual safety logic

Every retrieved chunk is classified relative to the specific pair being
asked about — never in isolation:

| Classification | Meaning |
|---|---|
| **pair_specific_evidence** | The chunk's own source drug is one of the pair, AND its text also names the *other* drug (or a known synonym). The strongest signal this layer produces. |
| **class_level_evidence** | One target drug is named; the text separately mentions a drug *class* the other target drug belongs to (e.g. "NSAIDs"), without naming it directly. |
| **drug_specific_evidence** | Relevant to *one* of the two target drugs, with no textual connection — direct or class-level — to the other. |
| **general_label_evidence** | Retrieved (semantically similar to a query) but doesn't clearly name either target drug or its class. |

None of these — individually or combined — are ever read as "an
interaction was found". They describe how directly a piece of text
*relates textually* to the pair, nothing about clinical truth.

### Why a chunk's own source drug counts as an implicit mention

FDA labels routinely say "this drug" rather than repeating their own
name. So a chunk retrieved from Warfarin's own indexed evidence counts
as "mentioning Warfarin" via its stored `rxcui` provenance, even without
the literal word "warfarin" appearing. But `pair_specific_evidence`
still requires the *other* drug's name to appear in the text itself —
provenance alone never satisfies both sides of the pair.

## Why a similarity score still isn't clinical certainty

Every `PairEvidenceItem.distance` carries forward Phase 5's rule,
unchanged: it's a vector-space text-similarity signal, nothing else. It
is never converted into, displayed as, or used to imply:

- interaction probability
- severity
- clinical risk
- confidence that an interaction exists

This phase introduces no `retrieval_confidence` field at all, precisely
to avoid inviting the misreading the project brief warned against. If a
future phase adds one, it must describe *retrieval quality* only (e.g.
"how many queries agreed"), and must say so explicitly in its own field
description — never clinical certainty.

## Drug mention detection — and its real limits

`app/rag/entity_matching.py` does case-insensitive, whole-word string
matching against a drug's known name variants, plus a small lookup table
mapping common class terms (NSAID, SSRI, MAOI, etc.) to representative
member drugs. This is **not** a clinical entity-resolution system. Its
concrete, documented limitations:

- **No negation handling.** "No interaction has been observed with
  ibuprofen" registers as a mention of ibuprofen, identically to a
  sentence describing a real interaction. Verified directly by
  `test_find_mentions_does_not_understand_negation`.
- **No unknown-synonym handling.** Only names already known to RxNorm
  (Phase 2) and stored in SQLite are matchable — a regional brand name
  or research code RxNorm doesn't know about won't be found.
- **A small, illustrative, non-exhaustive class vocabulary.** Not
  derived from RxNorm's own class system (RxClass) or any curated
  source. Its *absence* of a match must never be read as "no class
  relationship exists" — only "this simple lookup didn't find one".

Because of these limits, mention detection is used only to **sort and
label already-retrieved evidence** for later (human or LLM) review —
never to assert an interaction on its own.

## Every field required for provenance is preserved

`PairEvidenceItem` carries the drug, RxCUI, label set ID, SPL version,
manufacturer, section name, section LOINC code, source URL, database
record ID, chunk ID, the retrieval quer(y/ies) that found it, and the
distance score — all traceable back through Phase 5's ChromaDB metadata
to Phase 4's SQLite rows and, from there, to the original DailyMed
document. Verified directly by
`test_pair_evidence_preserves_full_provenance`.

## The result structure

```
PairRetrievalResult
├── drug_a / drug_b        (DrugRef: input name, rxcui, resolved?, search terms used)
├── evidence_status         one of:
│                              pair_specific_evidence_found
│                              supporting_evidence_found
│                              insufficient_evidence
│                              drug_not_found
│                              invalid_input
│                              retrieval_error
├── pair_evidence           [PairEvidenceItem, ...] — ranked by distance
├── supporting_evidence     [PairEvidenceItem, ...] — ranked by classification strength, then distance
├── queries_used            every query text actually executed
├── limitations              always-attached standard caveats (see STANDARD_LIMITATIONS)
└── warnings                 situational notes (e.g. "drug X not found")
```

**"no interaction" is deliberately not a possible `evidence_status`
value** — verified by a static test on the schema itself
(`test_evidence_status_never_uses_no_interaction_as_a_value`). This
layer only ever reports what it did or didn't find; concluding that no
interaction exists is a much stronger claim this project has not
implemented and does not intend to make from retrieval alone.

## Multi-drug efficiency

For N drugs, `generate_unique_pairs()` produces C(N,2) pairs — 4 drugs
→ 6 pairs, 10 drugs → 45 pairs. That growth is inherent to "every unique
pair" and isn't reduced here. What Phase 6 *does* do efficiently:
`PairRetrievalEngine` caches drug-specific query results (keyed by rxcui
+ query text), so a drug appearing in several pairs has its "drug X
interactions"-type queries executed against ChromaDB only once, reused
for every pair containing it — verified by
`test_retrieve_multi_reuses_cached_drug_specific_queries`, which asserts
`engine.cache_hits > 0` after a 3-drug batch.

## Design boundary: Phase 6 works on already-ingested data

`PairRetrievalEngine.resolve_drug()` looks a drug up in the local SQLite
knowledge base (Phase 4) — it does **not** call RxNorm live. If a drug
hasn't been ingested yet (Phases 2-5), it's reported `drug_not_found`,
distinct from "we checked and found no evidence". This keeps Phase 6
fully testable with controlled fixtures (as required) and matches the
project's layered architecture: each phase builds on what the previous
one already made durable and local.

## What Phase 6 explicitly does NOT do

- No LLM, no natural-language explanation generation.
- No interaction determination — evidence is surfaced and classified,
  never judged.
- No severity, probability, or risk scoring of any kind.
- No autonomous clinical recommendation.

## Viva preparation

**"What is the difference between retrieval and interaction
detection?"** Retrieval finds text that's *textually/semantically
similar* to a query — a statistical, local computation with no
understanding of pharmacology. Interaction detection would require
actually reasoning about whether that text *asserts* a real clinical
relationship, which is a judgment call this project deliberately leaves
to a later phase (an LLM reading the retrieved evidence), and even then,
always with a citation back to the source — never to the retrieval
signal itself.

**"Why can't vector similarity determine whether two drugs interact?"**
A distance score only reflects how alike two pieces of text are in the
embedding model's learned representation of language — it has no notion
of clinical truth, negation, or causality. Two sentences can be highly
similar in *topic* while saying opposite things ("X increases bleeding
risk" vs. "X has not been shown to increase bleeding risk" are close
in vector space, but mean opposite things).

**"How does your system find evidence for a drug pair?"** It runs
several different queries — combining both drug names together, and
each drug individually — against the previously indexed FDA label
evidence for those two specific drugs, then merges, deduplicates, and
classifies what comes back by whether the text actually names both
drugs (or a class containing one of them), not just by how similar it
sounds to the query.

**"What happens if the evidence only discusses one drug?"** It's
classified `drug_specific_evidence` and placed in `supporting_evidence`,
never in `pair_evidence` — and the pair's overall status becomes
`supporting_evidence_found`, a distinctly weaker claim than
`pair_specific_evidence_found`.

**"What happens if the system cannot find pair-specific evidence?"**
If there's *some* relevant evidence about either drug individually, the
status is `supporting_evidence_found`. If there's genuinely nothing
retrievable for either drug, it's `insufficient_evidence`. Neither of
these — nor any other status this layer can produce — is ever reported
as "no interaction". The system simply says what it did or didn't find.

## Known limitations

1. Mention detection has no negation understanding (see above) —
   inherited into every classification this phase produces.
2. The drug-class vocabulary is small and illustrative, not clinically
   curated.
3. Evidence is limited to what's already been ingested (Phases 2-5) —
   this phase never triggers fresh RxNorm/DailyMed/embedding calls.
4. Pair-specific evidence requires the connection to appear within one
   of the two drugs' *own* FDA labels — a real interaction documented
   only in third-party literature, not mentioned in either drug's own
   label text, would not surface as pair-specific evidence here.
5. Query generation is bounded (max 2 name variants per drug) — a very
   obscure brand name beyond that cap wouldn't get its own dedicated
   query, though it could still surface via semantic similarity or
   provenance-based matching.
6. C(N,2) pair growth for multi-drug lists is real and unmitigated at
   the pair-count level (only per-drug query execution is cached).
