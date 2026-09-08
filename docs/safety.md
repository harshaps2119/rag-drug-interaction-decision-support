# Safety & Validation — Phase 7

This document explains the safety architecture around the LLM layer,
and gives prepared answers for viva questions.

## The one-sentence explanation

> **Gemini explains evidence that retrieval already found and classified.
> It cannot invent evidence, upgrade its strength, or claim an
> interaction that retrieval didn't establish — and if it tries, its
> entire response is discarded in favor of showing the evidence itself.**

## Why "insufficient evidence" is different from "no interaction"

This distinction has been enforced since Phase 3 and holds all the way
through this phase: **absence of evidence is not evidence of absence.**
Phase 3's DailyMed integration already distinguished "no interactions
section in this label" from "no interaction exists." Phase 6 carried
that forward: `PairEvidenceStatus` never includes a "no interaction"
value. This phase adds the LLM on top of that same principle, mechanically
enforced twice:

1. **The schema itself.** `LLMStructuredOutput.interaction_assessment`
   only accepts three values — `pair_specific_evidence_found`,
   `supporting_evidence_found`, `insufficient_evidence` — mirroring
   Phase 6 exactly. There is no code path through which the model can
   set this field to anything resembling "no interaction"; if it tries,
   JSON parsing/schema validation itself rejects the response.
2. **A phrase-level check on the free text.** Even if Gemini doesn't use
   that field, it might still write "these drugs do not interact" inside
   `evidence_summary` or another prose field. `grounding_validator.py`
   scans for exactly this class of phrase and treats any match as fatal.

Both are demonstrated directly in `tests/test_safety_regression.py`:
Test 4 confirms `insufficient_evidence` is reported correctly when
there's genuinely nothing retrieved, and Test 4b confirms that even a
misbehaving model explicitly writing "no interaction exists" gets its
entire response discarded, never shown to a user.

## Why unsupported severity is *corrected*, not *rejected wholesale*

This is a deliberate asymmetry, worth understanding for the viva. Four
of the five validator checks are **fatal** — any single failure discards
the *entire* response:

- a hallucinated evidence citation
- an upgraded evidence-strength claim
- "no interaction" language
- a fabricated URL

These are all **structural trust failures**. A model willing to invent
one citation, or upgrade class-level evidence to pair-specific, cannot
be trusted on anything else it said in that same response — there's no
principled way to keep "the good parts."

**Severity is handled differently — stripped, not fatal** — because it's
a narrower, more mechanically checkable, self-contained claim: does this
exact short phrase appear in the cited evidence text, yes or no? An
overreaching one-word severity guess doesn't cast the same doubt over
the rest of a response (an accurate evidence_summary correctly citing
real evidence) the way a fabricated citation does. Discarding an
otherwise-good, well-grounded explanation just because of one
unsupported adjective would make the system *less* useful without making
it meaningfully safer — the actually dangerous claim (an invented
severity shown to a user) is fully prevented either way, since the field
is nulled before display. See
`tests/test_safety_regression.py::test_5_unsupported_severity_claim_is_rejected_by_validator`
and `::test_5b_end_to_end_unsupported_severity_never_reaches_user` for
the executed proof of both halves of this claim: the bad severity never
reaches the user, and the rest of the response still does.

## The evidence hierarchy is never re-ranked by the LLM

Phase 6's four evidence classifications
(`pair_specific` > `class_level`/`drug_specific` > `general_label`) and
its ordinal ranking of the three possible pair-level statuses
(`insufficient_evidence` < `supporting_evidence_found` <
`pair_specific_evidence_found`) are treated as ground truth the LLM
layer can only ever *match or be more conservative than*, never exceed.
`grounding_validator._ASSESSMENT_RANK` encodes this ordering explicitly,
and `test_llm_claiming_supporting_when_retrieval_found_pair_specific_is_allowed`
proves the asymmetry directly: the model IS allowed to hedge more than
retrieval strictly requires (reporting `supporting_evidence_found` even
when pair-specific evidence exists is fine — being extra cautious is
never penalized), but
`test_llm_claiming_pair_specific_when_retrieval_only_found_supporting_is_fatal`
proves the reverse is always rejected.

## Evidence-only fallback: the actual safety net

Every failure mode this phase can encounter — missing/invalid API key,
timeout, rate limit, service unavailable, malformed JSON, or ANY fatal
validation issue — routes to the exact same
`explanation_service.build_evidence_only_response()`. This function:

- Shows every retrieved evidence item, fully cited, exactly as Phase 6
  classified it.
- Generates **no prose at all** — no summary, no clinical effect, no
  mechanism, no severity. Nothing invented fills the gap.
- Always includes a plain-language `fallback_reason` explaining exactly
  why the LLM path wasn't used.
- Always includes the standard safety notice.

This is the concrete implementation of the project's core safety
requirement: *a broken or untrustworthy LLM response must never silently
become an invented-looking answer.* The user always sees something real
and traceable — the actual FDA label text Phase 6 found — never a gap
papered over with a guess.

## Viva preparation

**"Why use an LLM at all if RAG already retrieves the evidence?"**
Retrieval finds and classifies relevant text — it does not produce
readable prose explaining it, and raw FDA label excerpts are often dense
and hard for a non-specialist to parse quickly. The LLM's entire job is
turning that already-verified evidence into a clear summary — never
adding new facts, never being the source of what counts as evidence.

**"How do you prevent hallucination?"** Layered, not single-point:
(1) the prompt explicitly forbids it and explains the evidence hierarchy
in plain terms; (2) the output schema is narrow, so there's no field the
model could use to smuggle in unstructured invented content; (3) the
grounding validator independently, mechanically re-checks every claim
against the actual evidence and the actual Phase 6 status, with zero
trust in the model's self-report; (4) any failure of that check discards
the whole response rather than trying to salvage it.

**"How do you validate an LLM response?"** Five checks, described above
and in `docs/llm.md`: evidence ids must exist, the assessment can't
exceed retrieval's own finding, no "no interaction" language, no
fabricated URLs (fatal, all four), and severity must be verbatim-present
in cited text (non-fatal, stripped if not).

**"Can Gemini determine drug interaction severity?"** No — and the
system is built so it structurally cannot. It may only *report* a
severity phrase if the retrieved evidence explicitly states one, and
even then, the validator independently confirms that exact phrase is
present in the cited text before it's shown. It can never estimate,
infer, or standardize a severity the source doesn't state.

**"What happens when the evidence is insufficient?"** The system reports
`insufficient_evidence` — explicitly not "no interaction" — and still
generates a short LLM explanation only if that's a truthful statement of
uncertainty; if the model tries to say more than that (e.g. asserts
safety), the response is discarded and the (empty) evidence-only view is
shown instead.

**"What happens if Gemini is unavailable?"** The system falls back to
showing exactly what Phase 6 retrieved — full evidence, full citations —
with no generated prose and a clear reason why. The application stays
fully usable; it just does less.

**"Why shouldn't the LLM generate its own citations?"** A hallucinated
`EVIDENCE-999` id is mechanically, unambiguously detectable — it's
simply not in the map the backend built from real, already-verified
data. A hallucinated URL or reference name would look exactly as
plausible as a real one, with no cheap way to tell the difference. Citing
by backend-controlled id, then letting the backend (not the model)
resolve that id to real source detail, removes the entire class of
"plausible-looking fake citation" risk.

**"What is the difference between evidence retrieval and clinical
reasoning?"** Retrieval and classification (Phases 5-6) are mechanical,
deterministic, and fully offline-testable — they never make a judgment
about clinical truth, only about textual/semantic relevance. Clinical
reasoning — actually deciding what a body of evidence *means* for a
patient — is not something this project claims to do at all. The LLM
layer explains retrieved text; it does not, and is actively prevented
from, reasoning to a clinical conclusion beyond what that text states.
