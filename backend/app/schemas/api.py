"""
schemas/api.py
=================
Request/response models for the FastAPI routers (Phase 8), including
the input validation rules for drug names.

VALIDATION RULES, DOCUMENTED
----------------------------------
A drug name (after stripping leading/trailing whitespace) must:
  - be non-empty
  - be at most settings.MAX_DRUG_NAME_LENGTH characters (default 200 --
    generous; real drug names, including long chemical/combination
    names, are nowhere near this long)
  - match ALLOWED_NAME_PATTERN: letters, digits, spaces, and the
    punctuation that legitimately appears in real drug names --
    hyphens, periods, commas, parentheses, slashes, and apostrophes.
    Examples this pattern correctly ALLOWS: "Co-trimoxazole",
    "Vitamin B-12", "Humalog 75/25", "Tylenol PM (Extra Strength)",
    "St. John's Wort", "5-fluorouracil".
    Examples this pattern correctly REJECTS: anything containing
    <, >, ;, {, }, `, or other characters with no legitimate place in a
    drug name and every reason to be suspicious (e.g. injection
    attempts) in a field that flows into database queries and an LLM
    prompt.

For the multi-drug endpoint, additionally:
  - at least 2 drugs, at most settings.MAX_DRUGS_PER_MULTI_REQUEST
    (default 10 -- C(10,2)=45 pairs, matching the scaling example
    already documented in docs/retrieval.md)
  - no duplicate names (case-insensitive) -- rejected with a clear
    error naming the duplicate, rather than silently deduplicated,
    since a caller who accidentally sent the same drug twice likely
    wants to know, not have it silently dropped.

None of this validation duplicates or weakens Phase 6/7's own handling
of "unknown drug" -- a syntactically valid but non-existent drug name
still passes this layer's validation and is reported downstream as
`drug_not_found`, exactly as before this phase.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.schemas.explanation import ExplanationResponse

ALLOWED_NAME_PATTERN = re.compile(r"^[A-Za-z0-9\s\-\.\,\(\)/'%]+$")


def _validate_single_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Drug name must not be empty.")
    if len(cleaned) > settings.MAX_DRUG_NAME_LENGTH:
        raise ValueError(f"Drug name exceeds the maximum length of {settings.MAX_DRUG_NAME_LENGTH} characters.")
    if not ALLOWED_NAME_PATTERN.match(cleaned):
        raise ValueError(
            "Drug name contains characters that are not allowed. Letters, numbers, spaces, and "
            "-.,()/'% are permitted."
        )
    return cleaned


class InteractionCheckRequest(BaseModel):
    drug_a: str = Field(..., description="First drug name.")
    drug_b: str = Field(..., description="Second drug name.")

    @field_validator("drug_a", "drug_b")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        return _validate_single_name(v)


class MultiDrugCheckRequest(BaseModel):
    drugs: list[str] = Field(..., description="List of drug names to check all unique pairs of.")

    @field_validator("drugs")
    @classmethod
    def _validate_drugs(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("At least two drug names are required.")
        if len(v) > settings.MAX_DRUGS_PER_MULTI_REQUEST:
            raise ValueError(
                f"Too many drugs: {len(v)} given, maximum is {settings.MAX_DRUGS_PER_MULTI_REQUEST}."
            )

        cleaned = [_validate_single_name(name) for name in v]

        seen_lower: dict[str, str] = {}
        duplicates: list[str] = []
        for name in cleaned:
            key = name.lower()
            if key in seen_lower:
                duplicates.append(name)
            else:
                seen_lower[key] = name
        if duplicates:
            raise ValueError(f"Duplicate drug name(s) in request: {', '.join(sorted(set(duplicates)))}.")

        return cleaned


class PairResultEntry(BaseModel):
    """One pair's result within a multi-drug response."""

    drug_a: str
    drug_b: str
    result: ExplanationResponse


class MultiDrugCheckResponse(BaseModel):
    pairs: list[PairResultEntry]
    total_pairs: int
    request_id: str


class DrugSearchResult(BaseModel):
    rxcui: str
    normalized_name: str
    term_type: str | None = None


class HealthCheckDetail(BaseModel):
    status: str = Field(description="'ok' or 'error'.")
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' if all checks passed, 'degraded' if any failed.")
    checks: dict[str, HealthCheckDetail]
    request_id: str
