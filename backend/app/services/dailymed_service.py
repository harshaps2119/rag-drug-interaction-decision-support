"""
services/dailymed_service.py
===============================
Talks to the public DailyMed REST API (v2) to find and retrieve FDA
Structured Product Labels (SPLs), and extract the specific sections
relevant to drug interaction checking.

WHAT WE ARE DOING
------------------
DailyMed is the NLM/FDA's official repository of drug labeling (package
inserts). Every approved US drug product has one or more SPL documents —
formal HL7-standard XML documents with clinically reviewed content on
interactions, contraindications, warnings, etc. This service finds the
right SPL(s) for a drug and pulls out just the sections we care about.

WHY WE ARE DOING IT THIS WAY
--------------------------------
1. SEARCH BY RXCUI FIRST. Phase 2 already normalized the user's drug name
   to an RxCUI. DailyMed's /spls.json search accepts an `rxcui` filter
   directly, which is far more precise than a free-text name search (it
   avoids pulling in unrelated products that merely share a word in their
   name). We fall back to a drug_name search only if no RxCUI is available
   or the RxCUI search returns nothing.

2. THE FULL LABEL CONTENT IS XML-ONLY. DailyMed's /spls.json (search) can
   return JSON, but the actual label CONTENT endpoint, /spls/{SETID}, only
   supports XML — there is no JSON option for the full document. So we
   parse HL7 SPL XML directly (stdlib xml.etree.ElementTree).

3. WE NEVER ASSUME A SECTION EXISTS. SPL authors are only required to fill
   in certain sections, and even when a section is nominally present it is
   sometimes just a placeholder (we've seen FDA's own SPL authoring
   guidance note that authors should enter a single "." if a section has
   "no content to add" — so a lone "." is treated as NOT meaningful
   content here, not as evidence). Every target section we look for is
   returned in the result explicitly marked found=True/False, never
   silently omitted.

4. WE CHECK UP TO 3 CANDIDATE LABELS PER DRUG. In practice, DailyMed often
   has many different SPLs for the "same" drug (the brand manufacturer,
   plus several generic repackagers), and some of those SPLs have sparse
   or missing sections while others (usually the original/brand label)
   are complete. We check up to MAX_CANDIDATES_TO_CHECK candidates and use
   the first one that actually has interaction evidence, falling back to
   the first candidate's data if none do. This is documented as a known
   limitation, not hidden.

5. WE NEVER INFER SEVERITY. This module extracts TEXT ONLY. No keyword
   matching, scoring, or classification of severity happens here or
   anywhere in this phase — that would risk fabricating clinical judgment
   the source doesn't actually contain.

OFFICIAL LOINC SECTION CODES USED (verified against FDA's "Section
Headings (LOINC)" reference, https://www.fda.gov/industry/structured-
product-labeling/section-headings-loinc)
------------------------------------------------------------------------
    34073-7   Drug Interactions
    34074-5   Drug and/or Laboratory Test Interactions
    34070-3   Contraindications
    43685-7   Warnings and Precautions
    34084-4   Adverse Reactions
    34090-1   Clinical Pharmacology
    34066-1   Boxed Warning

WHAT THE CODE DOES (key functions)
------------------------------------
- get_drug_label_info(rxcui, drug_name) -> the main entry point.
- _search_spls()                         -> /spls.json search (by rxcui or name)
- _fetch_label_xml()                     -> /spls/{setid}.xml raw fetch
- _parse_label_xml()                     -> extracts doc metadata + target sections
- _extract_section_text()                -> flattens one <section>'s <text> to plain text
- _is_meaningful_text()                  -> filters out empty/placeholder content

HOW TO TEST IT (mocked, runs offline)
----------------------------------------
    cd backend
    pytest tests/test_dailymed_service.py -v

HOW TO TEST IT AGAINST THE REAL LIVE API (run this yourself — this
sandbox's network egress blocks dailymed.nlm.nih.gov, same as RxNorm;
see scripts/test_dailymed_live.py)
    cd backend
    python ../scripts/test_dailymed_live.py
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.exceptions import (
    ExternalAPIBadResponseError,
    ExternalAPITimeoutError,
    ExternalAPIUnavailableError,
)
from app.schemas.label import DrugLabelResult, LabelSection, SPLSummary

HL7_NS = "urn:hl7-org:v3"
NS = {"v3": HL7_NS}

MAX_CANDIDATES_TO_CHECK = 3
MAX_SEARCH_RESULTS = 10

# Internal key -> (LOINC section code, human-readable name)
TARGET_SECTIONS: dict[str, tuple[str, str]] = {
    "drug_interactions": ("34073-7", "Drug Interactions"),
    "drug_lab_interactions": ("34074-5", "Drug and/or Laboratory Test Interactions"),
    "contraindications": ("34070-3", "Contraindications"),
    "warnings_and_precautions": ("43685-7", "Warnings and Precautions"),
    "adverse_reactions": ("34084-4", "Adverse Reactions"),
    "clinical_pharmacology": ("34090-1", "Clinical Pharmacology"),
    "boxed_warning": ("34066-1", "Boxed Warning"),
}

# The set of internal keys whose presence (with real content) counts as
# "we found interaction evidence".
INTERACTION_SECTION_KEYS = ("drug_interactions", "drug_lab_interactions")

_RETRYABLE_EXCEPTIONS = (httpx.TransportError, httpx.TimeoutException)

# SPL authoring guidance instructs authors to fill empty sections with a
# single "." — these placeholder values must NOT be treated as content.
_PLACEHOLDER_VALUES = {"", ".", "-", "n/a", "na", "none", "not applicable"}


def _client() -> httpx.Client:
    return httpx.Client(base_url=settings.DAILYMED_BASE_URL, timeout=15.0)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
def _raw_get(client: httpx.Client, path: str, params: dict | None = None) -> httpx.Response:
    """Retry-wrapped raw HTTP call. See rxnorm_service._raw_get for why
    translation happens one layer up, not inside the retried function."""
    return client.get(path, params=params)


def _get_json(client: httpx.Client, path: str, params: dict | None = None) -> dict:
    """Shared JSON GET wrapper (used for the /spls.json search endpoint)."""
    response = _safe_get(client, path, params)
    if response.status_code == 404:
        # DailyMed returns 404 for "no results" on some search combinations.
        return {"data": []}
    _raise_for_status(response, path)
    try:
        return response.json()
    except ValueError as exc:
        raise ExternalAPIBadResponseError(
            f"DailyMed API returned non-JSON response for {path}", source="DailyMed"
        ) from exc


def _get_xml(client: httpx.Client, path: str) -> str:
    """Shared XML GET wrapper (used for the /spls/{setid}.xml full-label endpoint)."""
    response = _safe_get(client, path, params=None)
    _raise_for_status(response, path)
    return response.text


def _safe_get(client: httpx.Client, path: str, params: dict | None) -> httpx.Response:
    try:
        return _raw_get(client, path, params=params)
    except httpx.TimeoutException as exc:
        raise ExternalAPITimeoutError(
            f"DailyMed API timed out calling {path}", source="DailyMed"
        ) from exc
    except httpx.TransportError as exc:
        raise ExternalAPIUnavailableError(
            f"DailyMed API unreachable calling {path}: {exc}", source="DailyMed"
        ) from exc


def _raise_for_status(response: httpx.Response, path: str) -> None:
    if response.status_code >= 500:
        raise ExternalAPIUnavailableError(
            f"DailyMed API returned {response.status_code} for {path}",
            source="DailyMed",
            status_code=response.status_code,
        )
    if response.status_code >= 400 and response.status_code != 404:
        raise ExternalAPIBadResponseError(
            f"DailyMed API returned {response.status_code} for {path}: {response.text[:200]}",
            source="DailyMed",
            status_code=response.status_code,
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _search_spls(client: httpx.Client, *, rxcui: str | None = None, drug_name: str | None = None) -> list[SPLSummary]:
    """
    Search /spls.json by rxcui (preferred) or drug_name (fallback).
    Returns an empty list if nothing matches — this is a normal, expected
    outcome, not an error.
    """
    params: dict = {"pagesize": MAX_SEARCH_RESULTS}
    if rxcui:
        params["rxcui"] = rxcui
    elif drug_name:
        params["drug_name"] = drug_name
    else:
        return []

    data = _get_json(client, "/spls.json", params=params)
    rows = data.get("data", []) or []

    results = []
    for row in rows:
        setid = row.get("setid")
        title = row.get("title")
        if not setid or not title:
            continue
        results.append(
            SPLSummary(
                setid=setid,
                spl_version=str(row["spl_version"]) if row.get("spl_version") is not None else None,
                title=title,
                published_date=row.get("published_date"),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Full label fetch + XML parsing
# ---------------------------------------------------------------------------

def _fetch_label_xml(client: httpx.Client, setid: str) -> str:
    return _get_xml(client, f"/spls/{setid}.xml")


def _is_meaningful_text(text: str) -> bool:
    """
    Returns False for empty strings and known SPL placeholder values
    (FDA SPL authoring guidance permits a lone "." to mean "no content").
    """
    return text.strip().lower() not in _PLACEHOLDER_VALUES


def _extract_section_text(section_elem: ET.Element) -> str:
    """
    Flattens a <section>'s direct <text> block (paragraphs, lists, tables,
    etc.) into plain whitespace-normalized text.

    NOTE: This intentionally produces FLAT text, not structured
    paragraphs/lists. That's sufficient for Phase 3 (evidence retrieval);
    Phase 5 (chunking for the vector store) is where we will decide how
    much structure to preserve for embeddings.
    """
    text_elem = section_elem.find(f"{{{HL7_NS}}}text")
    if text_elem is None:
        return ""
    raw = " ".join(text_elem.itertext())
    return re.sub(r"\s+", " ", raw).strip()


def _find_target_sections(root: ET.Element) -> dict[str, LabelSection]:
    """
    Walks ALL <section> elements anywhere in the document (sections can be
    nested inside other sections as subsections) and extracts the ones
    matching our TARGET_SECTIONS LOINC codes.

    If the same LOINC code appears more than once (rare, but SPLs can be
    messy), the first occurrence with meaningful text wins.
    """
    found_by_code: dict[str, str] = {}  # loinc code -> extracted text

    for section in root.iter(f"{{{HL7_NS}}}section"):
        code_elem = section.find(f"{{{HL7_NS}}}code")
        if code_elem is None:
            continue
        code = code_elem.get("code")
        if not code:
            continue

        # Only bother extracting text for codes we actually care about.
        wanted_codes = {c for c, _ in TARGET_SECTIONS.values()}
        if code not in wanted_codes:
            continue

        text = _extract_section_text(section)
        if code not in found_by_code or (
            not _is_meaningful_text(found_by_code[code]) and _is_meaningful_text(text)
        ):
            found_by_code[code] = text

    sections: dict[str, LabelSection] = {}
    for key, (code, name) in TARGET_SECTIONS.items():
        text = found_by_code.get(code, "")
        meaningful = _is_meaningful_text(text)
        sections[key] = LabelSection(
            section_code=code,
            section_name=name,
            text=text if meaningful else "",
            found=meaningful,
        )
    return sections


def _extract_manufacturer(root: ET.Element) -> str | None:
    """
    Extracts the labeler/manufacturer organization name from the SPL's
    <author> block, if present. SPL structure:
        <author><assignedEntity><representedOrganization><name>...
    Defensive against any of these being absent — returns None rather
    than raising, since manufacturer is documented as "where available".
    """
    author_elem = root.find(f"{{{HL7_NS}}}author")
    if author_elem is None:
        return None
    name_elem = author_elem.find(
        f".//{{{HL7_NS}}}representedOrganization/{{{HL7_NS}}}name"
    )
    if name_elem is None or not name_elem.text:
        return None
    return name_elem.text.strip()


def _parse_label_xml(xml_text: str) -> dict:
    """
    Parses a full SPL XML document and returns doc-level metadata plus
    extracted target sections.

    Raises ExternalAPIBadResponseError if the XML cannot be parsed at all
    (this is a real failure — different from "sections not found", which
    is a normal, expected outcome represented via found=False).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ExternalAPIBadResponseError(
            f"DailyMed returned unparseable XML: {exc}", source="DailyMed"
        ) from exc

    title_elem = root.find(f"{{{HL7_NS}}}title")
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else None

    setid_elem = root.find(f"{{{HL7_NS}}}setId")
    setid = setid_elem.get("root") if setid_elem is not None else None

    version_elem = root.find(f"{{{HL7_NS}}}versionNumber")
    version = version_elem.get("value") if version_elem is not None else None

    effective_time_elem = root.find(f"{{{HL7_NS}}}effectiveTime")
    effective_time = effective_time_elem.get("value") if effective_time_elem is not None else None

    manufacturer = _extract_manufacturer(root)

    sections = _find_target_sections(root)

    return {
        "title": title,
        "setid": setid,
        "spl_version": version,
        "effective_time": effective_time,
        "manufacturer": manufacturer,
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_drug_label_info(rxcui: str | None = None, drug_name: str | None = None) -> DrugLabelResult:
    """
    Main entry point. Given an RxCUI (preferred, from Phase 2) and/or a
    drug name, finds the most useful available SPL and extracts the
    sections relevant to drug interaction checking.

    Like rxnorm_service.normalize_drug_name(), this function does NOT
    raise for "no label found" (an expected, common outcome) but DOES
    catch and translate external API failures into evidence_status="error"
    so callers can distinguish "DailyMed has nothing" from "DailyMed is
    broken right now".
    """
    if not rxcui and not drug_name:
        return DrugLabelResult(
            evidence_status="error",
            error="No rxcui or drug_name supplied.",
            warnings=["Cannot search DailyMed without at least one identifier."],
        )

    try:
        with _client() as client:
            candidates: list[SPLSummary] = []
            if rxcui:
                candidates = _search_spls(client, rxcui=rxcui)
            if not candidates and drug_name:
                candidates = _search_spls(client, drug_name=drug_name)

            if not candidates:
                return DrugLabelResult(
                    query_drug_name=drug_name,
                    query_rxcui=rxcui,
                    evidence_status="no_label_found",
                    warnings=[
                        "No SPL (FDA label) found in DailyMed for this drug, by RxCUI or name."
                    ],
                )

            total_found = len(candidates)
            checked = 0
            first_parsed: dict | None = None
            first_summary: SPLSummary | None = None

            for summary in candidates[:MAX_CANDIDATES_TO_CHECK]:
                checked += 1
                xml_text = _fetch_label_xml(client, summary.setid)
                parsed = _parse_label_xml(xml_text)

                if first_parsed is None:
                    first_parsed = parsed
                    first_summary = summary

                interaction_found = any(
                    parsed["sections"][key].found for key in INTERACTION_SECTION_KEYS
                )
                if interaction_found:
                    return _build_result(
                        rxcui, drug_name, summary, parsed, checked, total_found,
                        evidence_status="interaction_evidence_found",
                    )

            # None of the checked candidates had interaction evidence —
            # return the first candidate's data with an honest status.
            assert first_parsed is not None and first_summary is not None
            return _build_result(
                rxcui, drug_name, first_summary, first_parsed, checked, total_found,
                evidence_status="label_found_no_interaction_section",
                extra_warning=(
                    f"Checked {checked} of {total_found} available label(s) for this drug; "
                    "none had a populated Drug Interactions section."
                ),
            )

    except (ExternalAPITimeoutError, ExternalAPIUnavailableError, ExternalAPIBadResponseError) as exc:
        return DrugLabelResult(
            query_drug_name=drug_name,
            query_rxcui=rxcui,
            evidence_status="error",
            error=str(exc),
            warnings=[
                "DailyMed lookup service failed. This does NOT mean no label/interaction "
                "information exists — it means we could not check right now."
            ],
        )


def _build_result(
    rxcui: str | None,
    drug_name: str | None,
    summary: SPLSummary,
    parsed: dict,
    checked: int,
    total_found: int,
    *,
    evidence_status: str,
    extra_warning: str | None = None,
) -> DrugLabelResult:
    warnings = [extra_warning] if extra_warning else []
    return DrugLabelResult(
        query_drug_name=drug_name,
        query_rxcui=rxcui,
        evidence_status=evidence_status,  # type: ignore[arg-type]
        setid=parsed.get("setid") or summary.setid,
        spl_version=parsed.get("spl_version") or summary.spl_version,
        title=parsed.get("title") or summary.title,
        manufacturer=parsed.get("manufacturer"),
        published_date=summary.published_date,
        source_url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={summary.setid}",
        api_url=f"{settings.DAILYMED_BASE_URL}/spls/{summary.setid}.xml",
        sections=parsed["sections"],
        candidates_checked=checked,
        total_candidates_found=total_found,
        warnings=warnings,
    )
