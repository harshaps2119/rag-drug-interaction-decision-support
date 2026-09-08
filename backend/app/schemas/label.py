"""
schemas/label.py
==================
Pydantic models describing DailyMed / SPL (Structured Product Label) data
as it flows from the DailyMed service to the rest of the system.

KEY CONCEPT: evidence_status
-------------------------------
Per the project's core safety requirement, we must never conflate these
three genuinely different situations:

  1. "interaction_evidence_found"        -> We found a label, AND its
     Drug Interactions section has real content.
  2. "label_found_no_interaction_section" -> We found a label for this
     drug, but it has no (or only placeholder/empty) Drug Interactions
     section. This is common — many SPLs, especially for generic/OTC
     repackaged products, don't fill this section out.
  3. "no_label_found"                     -> DailyMed has no SPL at all
     for this drug (by RxCUI or name). This says nothing about whether
     an interaction exists — just that we have no FDA label to check.
  4. "error"                               -> The DailyMed API itself
     failed. Different again from "no label" — the service being down
     is not evidence of anything about the drug.

This distinction is what lets the downstream RAG/LLM layer (Phase 6-8)
honestly say "no interaction identified in the searched sources" instead
of the dangerous "no interaction exists".
"""

from typing import Literal

from pydantic import BaseModel, Field

EvidenceStatus = Literal[
    "interaction_evidence_found",
    "label_found_no_interaction_section",
    "no_label_found",
    "error",
]


class SPLSummary(BaseModel):
    """One row from a DailyMed /spls.json search result — metadata only, no content."""

    setid: str
    spl_version: str | None = None
    title: str
    published_date: str | None = None


class LabelSection(BaseModel):
    """
    One extracted section of an SPL label (e.g. Drug Interactions).

    `found=False` with empty text means: we looked for this section by its
    official LOINC code and it was genuinely absent (or contained only
    placeholder content like a lone "."), NOT that we failed to look.
    """

    section_code: str = Field(description="Official FDA/LOINC section code, e.g. '34073-7'.")
    section_name: str = Field(description="Human-readable section name.")
    text: str = Field(default="", description="Extracted plain text of the section, if present.")
    found: bool = Field(description="Whether this section was present with meaningful content.")


class DrugLabelResult(BaseModel):
    """
    The result of looking up DailyMed label information for one drug.
    This is the primary object returned by dailymed_service.get_drug_label_info().
    """

    query_drug_name: str | None = None
    query_rxcui: str | None = None

    evidence_status: EvidenceStatus

    setid: str | None = Field(default=None, description="DailyMed Set ID of the label actually used.")
    spl_version: str | None = None
    title: str | None = None
    manufacturer: str | None = Field(
        default=None, description="Labeler/manufacturer organization name, if present in the SPL."
    )
    published_date: str | None = None

    source_url: str | None = Field(
        default=None, description="Human-viewable DailyMed page for this label."
    )
    api_url: str | None = Field(
        default=None, description="Raw DailyMed API XML URL used to fetch this label's content."
    )

    sections: dict[str, LabelSection] = Field(
        default_factory=dict,
        description="Keyed by internal section key (e.g. 'drug_interactions', 'contraindications').",
    )

    candidates_checked: int = Field(
        default=0,
        description=(
            "How many SPL candidates (from possibly several manufacturers/repackagers "
            "of the same drug) were fetched and inspected before settling on this result."
        ),
    )
    total_candidates_found: int = Field(
        default=0, description="Total number of SPLs DailyMed returned for this query, before filtering."
    )

    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    def has_interaction_evidence(self) -> bool:
        return self.evidence_status == "interaction_evidence_found"
