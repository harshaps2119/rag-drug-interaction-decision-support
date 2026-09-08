"""
services/rxnorm_service.py
============================
Talks to the public RxNorm REST API to turn a free-text drug name (which
may be a generic name, a brand name, a synonym, or even a misspelling)
into a normalized RxNorm concept (RxCUI) that the rest of the system can
work with reliably.

WHAT WE ARE DOING
------------------
Drug names are messy: "Tylenol", "acetaminophen", "paracetamol",
"acetominophen" (typo), and "Tylenol Extra Strength" can all refer to
overlapping-but-different things. Before we can look up interactions, we
need ONE canonical identifier (the RxCUI) per drug. RxNorm is the NLM's
standard vocabulary for exactly this purpose.

WHY WE ARE DOING IT THIS WAY (3-tier lookup)
-----------------------------------------------
1. Try an EXACT/normalized match first (`/rxcui.json?search=2`). This
   endpoint does case-insensitive, punctuation-tolerant matching and is
   fast and precise — no guessing involved.
2. If that fails, fall back to RxNorm's APPROXIMATE match API
   (`/approximateTerm.json`), which is designed for misspellings
   ("worfarin" -> "warfarin"). We keep the confidence score and flag the
   result so the caller/UI can show it as a suggestion, not a certainty.
3. If nothing is found either way, we return match_type="none" rather than
   guessing — a wrong drug identity is dangerous in this domain.

Once we have an RxCUI, we fetch related concepts (`/allrelated.json`) to
populate brand names / generic name / synonyms, which the UI and later the
DailyMed lookup (Phase 3) will use.

WHAT THE CODE DOES (key functions)
------------------------------------
- normalize_drug_name(name)  -> the main entry point, returns a
  DrugNormalizationResult. Never raises for "not found" — only raises
  for the caller to handle if you use the lower-level functions directly.
- _find_exact_rxcui()         -> tier 1 lookup
- _find_approximate_matches() -> tier 2 lookup
- _get_related_concepts()     -> brand/generic/synonym enrichment
- _get_json()                 -> shared HTTP wrapper with retries + error
  translation into our custom exceptions (see app/exceptions.py)

WHAT YOU SHOULD UNDERSTAND
-----------------------------
1. RxCUI = RxNorm Concept Unique Identifier — the stable ID we key
   everything else off of (interaction lookup, label retrieval, etc.)
2. TTY (Term Type) tells you WHAT KIND of concept a name is: IN =
   Ingredient, BN = Brand Name, SCD = Semantic Clinical Drug (generic
   product), SBD = Semantic Branded Drug (branded product).
3. "Exact match" in RxNorm still tolerates case and minor punctuation —
   it is not literally byte-for-byte string equality.
4. Approximate match returns a SCORE (roughly 0-100). We only accept
   candidates above APPROX_MATCH_MIN_SCORE; below that we treat it as
   not found rather than silently picking a bad guess.
5. Network/API failures are modeled as a DIFFERENT outcome than "drug not
   found" — see match_type="error" in the schema docstring.

HOW TO TEST IT (mocked, runs offline)
----------------------------------------
    cd backend
    pytest tests/test_rxnorm_service.py -v

HOW TO TEST IT AGAINST THE REAL LIVE API (run this yourself — this
sandbox's network egress blocks rxnav.nlm.nih.gov, so it CANNOT be
executed from here; see scripts/test_rxnorm_live.py)
    cd backend
    python ../scripts/test_rxnorm_live.py
"""

from __future__ import annotations

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
from app.schemas.drug import DrugNormalizationResult, RxNormCandidate

APPROX_MATCH_MIN_SCORE = 50.0  # RxNorm scores are roughly 0-100; below this we don't trust the guess
MAX_APPROX_CANDIDATES = 5

# TTY codes we treat as "brand name" vs "ingredient/generic" when parsing allrelated.json
BRAND_TTYS = {"BN", "SBD", "SBDC", "SBDF", "SBDG"}
GENERIC_TTYS = {"IN", "PIN"}
SYNONYM_TTYS = {"SCD", "SCDC", "SCDF", "SCDG", "GPCK", "BPCK"}

_RETRYABLE_EXCEPTIONS = (httpx.TransportError, httpx.TimeoutException)


def _client() -> httpx.Client:
    return httpx.Client(base_url=settings.RXNORM_BASE_URL, timeout=10.0)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
def _raw_get(client: httpx.Client, path: str, params: dict | None = None) -> httpx.Response:
    """
    Retry-wrapped raw HTTP call. Deliberately does NOT catch or translate
    exceptions — tenacity's `retry_if_exception_type` needs to see the
    ORIGINAL httpx exception types (TimeoutException, TransportError) to
    decide whether to retry. If we translated them to our own exception
    types inside this function, the retry decorator would never recognize
    them as retryable and would give up after one attempt. Translation to
    our own exception types happens one layer up, in _get_json(), AFTER
    retries are exhausted.
    """
    return client.get(path, params=params)


def _get_json(client: httpx.Client, path: str, params: dict | None = None) -> dict:
    """
    Shared HTTP GET wrapper for all RxNorm calls.

    Retries transient network errors (timeouts, connection drops) up to 3
    times with exponential backoff (via _raw_get). Does NOT retry on 4xx
    errors (those are our fault — bad request — not transient), and
    translates all failures into our own exception types so callers don't
    need to know about httpx internals.
    """
    try:
        response = _raw_get(client, path, params=params)
    except httpx.TimeoutException as exc:
        raise ExternalAPITimeoutError(
            f"RxNorm API timed out calling {path}", source="RxNorm"
        ) from exc
    except httpx.TransportError as exc:
        raise ExternalAPIUnavailableError(
            f"RxNorm API unreachable calling {path}: {exc}", source="RxNorm"
        ) from exc

    if response.status_code >= 500:
        raise ExternalAPIUnavailableError(
            f"RxNorm API returned {response.status_code} for {path}",
            source="RxNorm",
            status_code=response.status_code,
        )
    if response.status_code >= 400:
        raise ExternalAPIBadResponseError(
            f"RxNorm API returned {response.status_code} for {path}: {response.text[:200]}",
            source="RxNorm",
            status_code=response.status_code,
        )

    try:
        return response.json()
    except ValueError as exc:
        raise ExternalAPIBadResponseError(
            f"RxNorm API returned non-JSON response for {path}", source="RxNorm"
        ) from exc


def _find_exact_rxcui(client: httpx.Client, name: str) -> str | None:
    """
    Tier 1: normalized/exact lookup via /rxcui.json?search=2.
    search=2 = "normalized" match: case-insensitive, tolerant of minor
    punctuation/whitespace differences, but NOT a fuzzy/spelling-correction
    match (that's tier 2).
    Returns the first RxCUI found, or None.
    """
    data = _get_json(client, "/rxcui.json", params={"name": name, "search": 2})
    id_group = data.get("idGroup", {})
    rxcui_list = id_group.get("rxnormId", [])
    return rxcui_list[0] if rxcui_list else None


def _find_approximate_matches(client: httpx.Client, name: str) -> list[RxNormCandidate]:
    """
    Tier 2: fuzzy match via /approximateTerm.json, for misspellings or
    near-matches. Returns candidates sorted by RxNorm's own ranking,
    already filtered to APPROX_MATCH_MIN_SCORE and capped at
    MAX_APPROX_CANDIDATES.
    """
    data = _get_json(
        client,
        "/approximateTerm.json",
        params={"term": name, "maxEntries": MAX_APPROX_CANDIDATES},
    )
    group = data.get("approximateGroup", {})
    raw_candidates = group.get("candidate", []) or []

    candidates: list[RxNormCandidate] = []
    for c in raw_candidates:
        rxcui = c.get("rxcui")
        score_str = c.get("score")
        if not rxcui or score_str is None:
            continue
        try:
            score = float(score_str)
        except (TypeError, ValueError):
            continue
        if score < APPROX_MATCH_MIN_SCORE:
            continue
        candidates.append(RxNormCandidate(rxcui=rxcui, name="", score=score))

    return candidates


def _get_canonical_name(client: httpx.Client, rxcui: str) -> str | None:
    """Fetch RxNorm's canonical display name for a given RxCUI."""
    data = _get_json(
        client,
        f"/rxcui/{rxcui}/property.json",
        params={"propName": "RxNorm Name"},
    )
    group = data.get("propConceptGroup", {})
    props = group.get("propConcept", []) or []
    if props:
        return props[0].get("propValue")
    return None


def _get_related_concepts(client: httpx.Client, rxcui: str) -> dict:
    """
    Tier: enrichment. Fetches /allrelated.json for the given RxCUI and
    extracts:
      - term_type: the TTY of the primary concept itself
      - brand_names: names from BN/SBD*-type concept groups
      - generic_name: the first IN/PIN concept name found
      - synonyms: names from SCD*-type concept groups (other clinical drug forms)

    Returns a plain dict (not a Pydantic model) since this is an internal
    helper feeding into the final DrugNormalizationResult.
    """
    data = _get_json(client, f"/rxcui/{rxcui}/allrelated.json")
    group = data.get("allRelatedGroup", {})
    concept_groups = group.get("conceptGroup", []) or []

    brand_names: list[str] = []
    generic_name: str | None = None
    synonyms: list[str] = []
    term_type: str | None = None

    for cg in concept_groups:
        tty = cg.get("tty")
        concept_props = cg.get("conceptProperties", []) or []

        # Does this concept group contain OUR rxcui? If so, that's our own term type.
        for cp in concept_props:
            if cp.get("rxcui") == rxcui:
                term_type = tty

        if tty in BRAND_TTYS:
            for cp in concept_props:
                cp_name = cp.get("name")
                if cp_name and cp_name not in brand_names:
                    brand_names.append(cp_name)
        elif tty in GENERIC_TTYS and generic_name is None:
            if concept_props:
                generic_name = concept_props[0].get("name")
        elif tty in SYNONYM_TTYS:
            for cp in concept_props:
                cp_name = cp.get("name")
                if cp_name and cp_name not in synonyms:
                    synonyms.append(cp_name)

    return {
        "term_type": term_type,
        "brand_names": brand_names[:10],   # cap for sane response sizes
        "generic_name": generic_name,
        "synonyms": synonyms[:10],
    }


def normalize_drug_name(name: str) -> DrugNormalizationResult:
    """
    Main entry point. Given a free-text drug name, returns a
    DrugNormalizationResult describing what RxNorm concept (if any) it
    resolves to.

    This function deliberately does NOT raise exceptions for "drug not
    found" (that's an expected, common outcome — see match_type="none").
    It DOES catch and translate external API failures into
    match_type="error" so the caller can distinguish "drug doesn't exist"
    from "the lookup service is broken right now" and message the user
    accordingly.
    """
    clean_name = name.strip()
    if not clean_name:
        return DrugNormalizationResult(
            input_name=name,
            match_type="none",
            warnings=["Empty drug name supplied."],
        )

    try:
        with _client() as client:
            rxcui = _find_exact_rxcui(client, clean_name)
            match_type = "exact"
            match_score = None
            other_candidates: list[RxNormCandidate] = []

            if rxcui is None:
                candidates = _find_approximate_matches(client, clean_name)
                if not candidates:
                    return DrugNormalizationResult(
                        input_name=clean_name,
                        match_type="none",
                        warnings=[
                            f"'{clean_name}' was not found in RxNorm, exactly or "
                            "approximately. Check spelling, or it may not be a "
                            "recognized US drug name."
                        ],
                    )
                best = candidates[0]
                rxcui = best.rxcui
                match_type = "approximate"
                match_score = best.score
                other_candidates = candidates[1:]

            canonical_name = _get_canonical_name(client, rxcui)
            related = _get_related_concepts(client, rxcui)

            warnings = []
            if match_type == "approximate":
                warnings.append(
                    f"'{clean_name}' did not match exactly. Using closest RxNorm "
                    f"match '{canonical_name}' (confidence score {match_score:.0f}/100). "
                    "Please confirm this is the intended drug."
                )

            return DrugNormalizationResult(
                input_name=clean_name,
                match_type=match_type,  # type: ignore[arg-type]
                rxcui=rxcui,
                normalized_name=canonical_name,
                term_type=related["term_type"],
                match_score=match_score,
                brand_names=related["brand_names"],
                generic_name=related["generic_name"],
                synonyms=related["synonyms"],
                other_candidates=other_candidates,
                source_url=f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={rxcui}",
                warnings=warnings,
            )

    except (ExternalAPITimeoutError, ExternalAPIUnavailableError, ExternalAPIBadResponseError) as exc:
        return DrugNormalizationResult(
            input_name=clean_name,
            match_type="error",
            error=str(exc),
            warnings=[
                "RxNorm lookup service failed. This does NOT mean the drug "
                "doesn't exist — it means we could not check right now."
            ],
        )
