"""
tests/test_api_validation.py
===============================
Tests for app/schemas/api.py's input validation — pure Pydantic model
tests, no HTTP/app needed for most of these, plus a few through the
real endpoint to confirm validation errors produce the standard error
envelope.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_api_validation.py -v
"""

import pytest
from pydantic import ValidationError

from app.schemas.api import InteractionCheckRequest, MultiDrugCheckRequest


# ---------------------------------------------------------------------------
# Legitimate drug names must NOT be over-restricted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "warfarin",
    "Co-trimoxazole",
    "Vitamin B-12",
    "Humalog 75/25",
    "Tylenol PM (Extra Strength)",
    "St. John's Wort",
    "5-fluorouracil",
    "Amoxicillin 500mg",
    "insulin glargine",
])
def test_legitimate_drug_names_are_accepted(name):
    req = InteractionCheckRequest(drug_a=name, drug_b="ibuprofen")
    assert req.drug_a == name


def test_leading_trailing_whitespace_is_stripped():
    req = InteractionCheckRequest(drug_a="  warfarin  ", drug_b="ibuprofen")
    assert req.drug_a == "warfarin"


# ---------------------------------------------------------------------------
# Empty / invalid names
# ---------------------------------------------------------------------------

def test_empty_drug_name_rejected():
    with pytest.raises(ValidationError):
        InteractionCheckRequest(drug_a="", drug_b="ibuprofen")


def test_whitespace_only_drug_name_rejected():
    with pytest.raises(ValidationError):
        InteractionCheckRequest(drug_a="   ", drug_b="ibuprofen")


def test_excessively_long_drug_name_rejected():
    with pytest.raises(ValidationError):
        InteractionCheckRequest(drug_a="a" * 500, drug_b="ibuprofen")


def test_name_at_exact_max_length_is_accepted():
    from app.config import settings

    name = "a" * settings.MAX_DRUG_NAME_LENGTH
    req = InteractionCheckRequest(drug_a=name, drug_b="ibuprofen")
    assert req.drug_a == name


@pytest.mark.parametrize("bad_name", [
    "warfarin<script>",
    "warfarin; DROP TABLE drugs;",
    "warfarin{{7*7}}",
    "warfarin`rm -rf`",
    "warfarin\x00nullbyte",
])
def test_invalid_characters_rejected(bad_name):
    with pytest.raises(ValidationError):
        InteractionCheckRequest(drug_a=bad_name, drug_b="ibuprofen")


# ---------------------------------------------------------------------------
# Multi-drug list validation
# ---------------------------------------------------------------------------

def test_multi_drug_minimum_two_required():
    with pytest.raises(ValidationError):
        MultiDrugCheckRequest(drugs=["warfarin"])


def test_multi_drug_empty_list_rejected():
    with pytest.raises(ValidationError):
        MultiDrugCheckRequest(drugs=[])


def test_multi_drug_valid_list_accepted():
    req = MultiDrugCheckRequest(drugs=["warfarin", "ibuprofen", "aspirin"])
    assert len(req.drugs) == 3


def test_multi_drug_too_many_rejected():
    from app.config import settings

    too_many = [f"drug{i}" for i in range(settings.MAX_DRUGS_PER_MULTI_REQUEST + 1)]
    with pytest.raises(ValidationError):
        MultiDrugCheckRequest(drugs=too_many)


def test_multi_drug_at_exact_max_is_accepted():
    from app.config import settings

    exactly_max = [f"drug{i}" for i in range(settings.MAX_DRUGS_PER_MULTI_REQUEST)]
    req = MultiDrugCheckRequest(drugs=exactly_max)
    assert len(req.drugs) == settings.MAX_DRUGS_PER_MULTI_REQUEST


def test_multi_drug_duplicate_names_rejected():
    with pytest.raises(ValidationError):
        MultiDrugCheckRequest(drugs=["warfarin", "ibuprofen", "Warfarin"])  # case-insensitive duplicate


def test_multi_drug_duplicate_error_names_the_duplicate():
    with pytest.raises(ValidationError) as exc_info:
        MultiDrugCheckRequest(drugs=["warfarin", "warfarin"])
    assert "warfarin" in str(exc_info.value).lower()


def test_multi_drug_one_invalid_name_rejects_whole_request():
    with pytest.raises(ValidationError):
        MultiDrugCheckRequest(drugs=["warfarin", "ibuprofen<script>"])


# ---------------------------------------------------------------------------
# Through the real endpoint: malformed JSON / validation errors -> standard envelope
# ---------------------------------------------------------------------------

def test_malformed_json_body_returns_standard_error_envelope(api_client):
    r = api_client.post(
        "/api/interaction/check", content=b"{not valid json", headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422
    body = r.json()
    assert set(body.keys()) == {"error"}
    assert body["error"]["code"] == "validation_error"
    assert "request_id" in body["error"]


def test_missing_required_field_returns_standard_error_envelope(api_client):
    r = api_client.post("/api/interaction/check", json={"drug_a": "warfarin"})  # missing drug_b
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_error"


def test_invalid_characters_through_endpoint_returns_422(api_client):
    r = api_client.post("/api/interaction/check", json={"drug_a": "warfarin<script>", "drug_b": "ibuprofen"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_validation_error_message_does_not_leak_internal_pydantic_paths(api_client):
    r = api_client.post("/api/interaction/check", json={"drug_a": ""})
    body = r.json()
    # Should not contain raw pydantic internals like "pydantic.errors" or a file path.
    assert "pydantic" not in body["error"]["message"].lower()
    assert "/app/" not in body["error"]["message"]
    assert "Traceback" not in body["error"]["message"]
