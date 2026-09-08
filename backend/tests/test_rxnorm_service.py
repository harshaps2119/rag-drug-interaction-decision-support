"""
tests/test_rxnorm_service.py
==============================
Unit tests for app/services/rxnorm_service.py.

WHY THESE TESTS MOCK THE HTTP LAYER
--------------------------------------
We use `respx` (an httpx-aware mocking library) to intercept HTTP calls
and return realistic, hand-crafted RxNorm API response shapes. This lets
us:
  1. Run tests instantly and offline (no dependency on RxNorm's uptime).
  2. Test error handling (timeouts, 500s, malformed JSON) that would be
     hard to reliably trigger against the real API on demand.
  3. Pin down exactly what a "known good" RxNorm response looks like, as
     a form of documentation.

These are NOT a substitute for testing against the real API — see
scripts/test_rxnorm_live.py for that. That script hits the real,
production RxNorm service and must be run by you locally, since this
sandbox's network egress blocks rxnav.nlm.nih.gov.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_rxnorm_service.py -v

EXPECTED RESULT
-----------------
All tests should PASS. If you break something in rxnorm_service.py,
the relevant test will fail and tell you which assumption broke.
"""

import httpx
import pytest
import respx

from app.services import rxnorm_service

BASE_URL = "https://rxnav.nlm.nih.gov/REST"


# ---------------------------------------------------------------------------
# Fixture-style helper response builders
# ---------------------------------------------------------------------------

def _rxcui_response(rxcui_list: list[str]) -> dict:
    return {"idGroup": {"name": "query", "rxnormId": rxcui_list}}


def _property_response(name: str) -> dict:
    return {
        "propConceptGroup": {
            "propConcept": [{"propName": "RxNorm Name", "propValue": name}]
        }
    }


def _allrelated_response(rxcui: str, own_tty: str, brand_names=None, generic_name=None, synonyms=None) -> dict:
    brand_names = brand_names or []
    synonyms = synonyms or []
    concept_groups = [
        {
            "tty": own_tty,
            "conceptProperties": [{"rxcui": rxcui, "name": "self", "tty": own_tty}],
        }
    ]
    if brand_names:
        concept_groups.append(
            {
                "tty": "BN",
                "conceptProperties": [{"rxcui": f"b{i}", "name": n, "tty": "BN"} for i, n in enumerate(brand_names)],
            }
        )
    if generic_name:
        concept_groups.append(
            {
                "tty": "IN",
                "conceptProperties": [{"rxcui": "g1", "name": generic_name, "tty": "IN"}],
            }
        )
    if synonyms:
        concept_groups.append(
            {
                "tty": "SCD",
                "conceptProperties": [{"rxcui": f"s{i}", "name": n, "tty": "SCD"} for i, n in enumerate(synonyms)],
            }
        )
    return {"allRelatedGroup": {"rxcui": rxcui, "conceptGroup": concept_groups}}


def _approximate_response(candidates: list[tuple[str, float]]) -> dict:
    return {
        "approximateGroup": {
            "inputTerm": "query",
            "candidate": [{"rxcui": rxcui, "score": str(score)} for rxcui, score in candidates],
        }
    }


# ---------------------------------------------------------------------------
# Tests: exact match (generic name)
# ---------------------------------------------------------------------------

@respx.mock
def test_normalize_exact_match_generic_name():
    """A well-known generic name like 'warfarin' should resolve on the first (exact) tier."""
    respx.get(f"{BASE_URL}/rxcui.json", params={"name": "warfarin", "search": "2"}).mock(
        return_value=httpx.Response(200, json=_rxcui_response(["11289"]))
    )
    respx.get(f"{BASE_URL}/rxcui/11289/property.json", params={"propName": "RxNorm Name"}).mock(
        return_value=httpx.Response(200, json=_property_response("warfarin"))
    )
    respx.get(f"{BASE_URL}/rxcui/11289/allrelated.json").mock(
        return_value=httpx.Response(
            200,
            json=_allrelated_response(
                "11289", own_tty="IN", brand_names=["Coumadin", "Jantoven"], synonyms=["warfarin 5 MG Oral Tablet"]
            ),
        )
    )

    result = rxnorm_service.normalize_drug_name("warfarin")

    assert result.match_type == "exact"
    assert result.rxcui == "11289"
    assert result.normalized_name == "warfarin"
    assert result.term_type == "IN"
    assert "Coumadin" in result.brand_names
    assert result.is_usable() is True
    assert result.error is None


# ---------------------------------------------------------------------------
# Tests: exact match (brand name)
# ---------------------------------------------------------------------------

@respx.mock
def test_normalize_exact_match_brand_name():
    """A brand name like 'Advil' should also resolve via the exact-match tier,
    and its generic ingredient (ibuprofen) should be populated."""
    respx.get(f"{BASE_URL}/rxcui.json", params={"name": "Advil", "search": "2"}).mock(
        return_value=httpx.Response(200, json=_rxcui_response(["153010"]))
    )
    respx.get(f"{BASE_URL}/rxcui/153010/property.json", params={"propName": "RxNorm Name"}).mock(
        return_value=httpx.Response(200, json=_property_response("Advil"))
    )
    respx.get(f"{BASE_URL}/rxcui/153010/allrelated.json").mock(
        return_value=httpx.Response(
            200,
            json=_allrelated_response("153010", own_tty="BN", generic_name="ibuprofen"),
        )
    )

    result = rxnorm_service.normalize_drug_name("Advil")

    assert result.match_type == "exact"
    assert result.term_type == "BN"
    assert result.generic_name == "ibuprofen"


# ---------------------------------------------------------------------------
# Tests: misspelling falls back to approximate match
# ---------------------------------------------------------------------------

@respx.mock
def test_normalize_misspelling_falls_back_to_approximate():
    """'worfarin' (typo) has no exact match, so we should fall back to the
    approximate-match tier and flag the result accordingly."""
    respx.get(f"{BASE_URL}/rxcui.json", params={"name": "worfarin", "search": "2"}).mock(
        return_value=httpx.Response(200, json=_rxcui_response([]))
    )
    respx.get(f"{BASE_URL}/approximateTerm.json", params={"term": "worfarin", "maxEntries": "5"}).mock(
        return_value=httpx.Response(200, json=_approximate_response([("11289", 90.0), ("999999", 55.0)]))
    )
    respx.get(f"{BASE_URL}/rxcui/11289/property.json", params={"propName": "RxNorm Name"}).mock(
        return_value=httpx.Response(200, json=_property_response("warfarin"))
    )
    respx.get(f"{BASE_URL}/rxcui/11289/allrelated.json").mock(
        return_value=httpx.Response(200, json=_allrelated_response("11289", own_tty="IN"))
    )

    result = rxnorm_service.normalize_drug_name("worfarin")

    assert result.match_type == "approximate"
    assert result.rxcui == "11289"
    assert result.match_score == 90.0
    assert result.normalized_name == "warfarin"
    assert len(result.other_candidates) == 1
    assert any("did not match exactly" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Tests: low-confidence approximate matches are rejected, not guessed
# ---------------------------------------------------------------------------

@respx.mock
def test_normalize_low_confidence_approximate_is_rejected():
    """If RxNorm's best fuzzy guess is below our confidence threshold, we
    must NOT silently accept it — safety requirement: no guessing."""
    respx.get(f"{BASE_URL}/rxcui.json", params={"name": "xyzzyx", "search": "2"}).mock(
        return_value=httpx.Response(200, json=_rxcui_response([]))
    )
    respx.get(f"{BASE_URL}/approximateTerm.json", params={"term": "xyzzyx", "maxEntries": "5"}).mock(
        return_value=httpx.Response(200, json=_approximate_response([("123", 10.0)]))
    )

    result = rxnorm_service.normalize_drug_name("xyzzyx")

    assert result.match_type == "none"
    assert result.rxcui is None
    assert result.is_usable() is False


# ---------------------------------------------------------------------------
# Tests: completely unknown drug
# ---------------------------------------------------------------------------

@respx.mock
def test_normalize_unknown_drug_returns_none():
    respx.get(f"{BASE_URL}/rxcui.json", params={"name": "notadrugname123", "search": "2"}).mock(
        return_value=httpx.Response(200, json=_rxcui_response([]))
    )
    respx.get(f"{BASE_URL}/approximateTerm.json", params={"term": "notadrugname123", "maxEntries": "5"}).mock(
        return_value=httpx.Response(200, json=_approximate_response([]))
    )

    result = rxnorm_service.normalize_drug_name("notadrugname123")

    assert result.match_type == "none"
    assert result.rxcui is None
    assert result.error is None
    assert len(result.warnings) == 1


# ---------------------------------------------------------------------------
# Tests: empty input handled without hitting the network at all
# ---------------------------------------------------------------------------

def test_normalize_empty_string_short_circuits():
    result = rxnorm_service.normalize_drug_name("   ")
    assert result.match_type == "none"
    assert "Empty drug name" in result.warnings[0]


# ---------------------------------------------------------------------------
# Tests: API failure is distinguished from "not found"
# ---------------------------------------------------------------------------

@respx.mock
def test_normalize_api_5xx_returns_error_not_none():
    """A 500 from RxNorm must surface as match_type='error', NEVER as
    match_type='none' — conflating 'service is down' with 'drug doesn't
    exist' would be a dangerous silent failure in a clinical tool."""
    respx.get(f"{BASE_URL}/rxcui.json", params={"name": "warfarin", "search": "2"}).mock(
        return_value=httpx.Response(503, text="Service Unavailable")
    )

    result = rxnorm_service.normalize_drug_name("warfarin")

    assert result.match_type == "error"
    assert result.error is not None
    assert result.rxcui is None


@respx.mock
def test_normalize_timeout_returns_error():
    respx.get(f"{BASE_URL}/rxcui.json", params={"name": "warfarin", "search": "2"}).mock(
        side_effect=httpx.TimeoutException("timed out")
    )

    result = rxnorm_service.normalize_drug_name("warfarin")

    assert result.match_type == "error"
    assert "timed out" in result.error.lower() or "timeout" in result.error.lower()


# ---------------------------------------------------------------------------
# Tests: low-level retry behavior
# ---------------------------------------------------------------------------

@respx.mock
def test_get_json_retries_on_transient_failure_then_succeeds():
    """The shared _get_json() wrapper should retry transient network errors
    and succeed if a later attempt works."""
    route = respx.get(f"{BASE_URL}/rxcui.json", params={"name": "warfarin", "search": "2"}).mock(
        side_effect=[
            httpx.TimeoutException("timed out"),
            httpx.Response(200, json=_rxcui_response(["11289"])),
        ]
    )

    with httpx.Client(base_url=BASE_URL) as client:
        data = rxnorm_service._get_json(client, "/rxcui.json", params={"name": "warfarin", "search": 2})

    assert data["idGroup"]["rxnormId"] == ["11289"]
    assert route.call_count == 2


@respx.mock
def test_get_json_does_not_retry_on_4xx():
    """4xx errors are our fault (bad request), not transient — must NOT retry."""
    route = respx.get(f"{BASE_URL}/rxcui.json", params={"name": "warfarin", "search": "2"}).mock(
        return_value=httpx.Response(400, text="Bad Request")
    )

    with httpx.Client(base_url=BASE_URL) as client:
        with pytest.raises(Exception):
            rxnorm_service._get_json(client, "/rxcui.json", params={"name": "warfarin", "search": 2})

    assert route.call_count == 1
