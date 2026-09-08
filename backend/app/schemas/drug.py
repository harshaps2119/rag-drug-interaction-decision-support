"""
schemas/drug.py
================
Pydantic models describing the SHAPE of drug-normalization data as it
flows between the RxNorm service, the API layer, and (later) the frontend.

WHY PYDANTIC MODELS HERE (not just dicts)
-------------------------------------------
1. Validation: if some code path forgets to set a field, or sets the wrong
   type, we find out immediately at construction time — not when the
   frontend crashes trying to read `undefined.rxcui`.
2. Self-documentation: anyone reading this file sees exactly what a
   "normalized drug" looks like without reverse-engineering it from code.
3. FastAPI (Phase 9) uses these same models to auto-generate OpenAPI/Swagger
   docs and validate request/response bodies for free.

KEY CONCEPT: match_type
-------------------------
RxNorm lookups can succeed in different ways, and the DIFFERENCE MATTERS
for a clinical-safety tool:
  - "exact"        -> the name matched an RxNorm concept directly/normalized.
  - "approximate"   -> no exact match; RxNorm's fuzzy/spelling-correction
                       API found a *likely* candidate (e.g. "worfarin" ->
                       "warfarin"). This should be shown to the user as a
                       suggestion, not silently treated as certain.
  - "none"          -> nothing found at all, exact or fuzzy.
  - "error"         -> the RxNorm API itself failed (network/service issue),
                       which is DIFFERENT from "drug doesn't exist". We must
                       never conflate "the API broke" with "the drug isn't
                       real" — that would be a dangerous silent failure mode.
"""

from typing import Literal

from pydantic import BaseModel, Field

MatchType = Literal["exact", "approximate", "none", "error"]


class RxNormCandidate(BaseModel):
    """A single fuzzy-match candidate returned by RxNorm's approximateTerm API."""

    rxcui: str
    name: str
    score: float = Field(description="RxNorm's match confidence score, roughly 0-100.")


class DrugNormalizationResult(BaseModel):
    """
    The result of trying to normalize one user-supplied drug name against
    RxNorm. This is the primary object returned by rxnorm_service.normalize_drug_name().
    """

    input_name: str = Field(description="Exactly what the user typed.")
    match_type: MatchType

    rxcui: str | None = Field(
        default=None, description="RxNorm Concept Unique Identifier, if resolved."
    )
    normalized_name: str | None = Field(
        default=None, description="RxNorm's canonical name for this concept."
    )
    term_type: str | None = Field(
        default=None,
        description=(
            "RxNorm TTY code for the matched concept, e.g. IN (Ingredient), "
            "BN (Brand Name), SCD (Semantic Clinical Drug), SBD (Semantic "
            "Branded Drug), PIN (Precise Ingredient)."
        ),
    )
    match_score: float | None = Field(
        default=None,
        description="Only set when match_type == 'approximate'. RxNorm's confidence score.",
    )

    brand_names: list[str] = Field(
        default_factory=list, description="Known brand names for this ingredient, if any."
    )
    generic_name: str | None = Field(
        default=None, description="The generic ingredient name, if the input was a brand name."
    )
    synonyms: list[str] = Field(
        default_factory=list, description="Other related RxNorm concept names (clinical drug forms etc.)."
    )

    other_candidates: list[RxNormCandidate] = Field(
        default_factory=list,
        description="Alternate fuzzy-match candidates when match_type == 'approximate', for transparency.",
    )

    source: str = "RxNorm"
    source_url: str | None = Field(
        default=None, description="Direct RxNav URL for this RxCUI, for user inspection."
    )

    warnings: list[str] = Field(default_factory=list)
    error: str | None = Field(
        default=None, description="Set only when match_type == 'error' (API failure, not 'not found')."
    )

    def is_usable(self) -> bool:
        """True if this result is reliable enough to proceed to interaction lookup."""
        return self.match_type in ("exact", "approximate") and self.rxcui is not None
