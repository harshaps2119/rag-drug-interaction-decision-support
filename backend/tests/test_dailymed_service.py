"""
tests/test_dailymed_service.py
=================================
Unit tests for app/services/dailymed_service.py.

Like test_rxnorm_service.py, these mock the HTTP layer with `respx` using
realistic response shapes based on DailyMed's real API structure (verified
against official NLM/FDA documentation — see the module docstring in
dailymed_service.py for sources). This lets us test parsing logic and
error handling deterministically and offline.

For a check against the REAL live API, see scripts/test_dailymed_live.py
(not executed by Claude — see that script's docstring for why).

HOW TO RUN
-----------
    cd backend
    pytest tests/test_dailymed_service.py -v

EXPECTED RESULT
-----------------
All tests should PASS.
"""

import httpx
import pytest
import respx

from app.services import dailymed_service

BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


# ---------------------------------------------------------------------------
# Minimal but structurally realistic SPL XML fixtures
# ---------------------------------------------------------------------------

def _spl_xml(
    setid: str,
    title: str,
    *,
    version: str = "3",
    effective_time: str = "20240115",
    interactions_text: str | None = "May increase INR when co-administered with NSAIDs such as ibuprofen.",
    contraindications_text: str | None = "Active pathological bleeding.",
    include_interactions_section: bool = True,
) -> str:
    """Builds a minimal, structurally-real HL7 SPL XML document for testing."""
    interactions_section = ""
    if include_interactions_section:
        text_content = interactions_text if interactions_text is not None else "."
        interactions_section = f"""
        <component>
          <section>
            <id root="a1111111-1111-1111-1111-111111111111"/>
            <code code="34073-7" codeSystem="2.16.840.1.113883.6.1" displayName="DRUG INTERACTIONS SECTION"/>
            <title>7 DRUG INTERACTIONS</title>
            <text>
              <paragraph>{text_content}</paragraph>
            </text>
          </section>
        </component>"""

    contraindications_section = ""
    if contraindications_text is not None:
        contraindications_section = f"""
        <component>
          <section>
            <id root="b2222222-2222-2222-2222-222222222222"/>
            <code code="34070-3" codeSystem="2.16.840.1.113883.6.1" displayName="CONTRAINDICATIONS SECTION"/>
            <title>4 CONTRAINDICATIONS</title>
            <text>
              <paragraph>{contraindications_text}</paragraph>
            </text>
          </section>
        </component>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="urn:hl7-org:v3">
  <id root="{setid}-doc"/>
  <setId root="{setid}"/>
  <versionNumber value="{version}"/>
  <title>{title}</title>
  <effectiveTime value="{effective_time}"/>
  <component>
    <structuredBody>
      {interactions_section}
      {contraindications_section}
    </structuredBody>
  </component>
</document>"""


def _search_response(rows: list[dict]) -> dict:
    return {
        "metadata": {"total_elements": str(len(rows))},
        "data": rows,
    }


# ---------------------------------------------------------------------------
# Tests: interaction evidence found on first candidate
# ---------------------------------------------------------------------------

@respx.mock
def test_get_label_info_interaction_found_by_rxcui():
    """Search by RxCUI (preferred, from Phase 2), single candidate, has interactions."""
    respx.get(f"{BASE_URL}/spls.json", params={"rxcui": "11289", "pagesize": "10"}).mock(
        return_value=httpx.Response(
            200,
            json=_search_response(
                [{"setid": "aaaa-1111", "spl_version": "3", "title": "WARFARIN SODIUM TABLET", "published_date": "Jan 15, 2024"}]
            ),
        )
    )
    respx.get(f"{BASE_URL}/spls/aaaa-1111.xml").mock(
        return_value=httpx.Response(
            200,
            text=_spl_xml("aaaa-1111", "WARFARIN SODIUM TABLET"),
            headers={"Content-Type": "application/xml"},
        )
    )

    result = dailymed_service.get_drug_label_info(rxcui="11289", drug_name="warfarin")

    assert result.evidence_status == "interaction_evidence_found"
    assert result.setid == "aaaa-1111"
    assert result.candidates_checked == 1
    assert result.total_candidates_found == 1
    assert result.sections["drug_interactions"].found is True
    assert "ibuprofen" in result.sections["drug_interactions"].text
    assert result.sections["contraindications"].found is True
    assert result.source_url == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=aaaa-1111"


# ---------------------------------------------------------------------------
# Tests: label found, but no interactions section anywhere -> honest status
# ---------------------------------------------------------------------------

@respx.mock
def test_get_label_info_label_found_no_interactions_section():
    """Two candidates, neither has a real Drug Interactions section — must
    NOT be reported as 'no interaction', only as 'section absent'."""
    respx.get(f"{BASE_URL}/spls.json", params={"rxcui": "999", "pagesize": "10"}).mock(
        return_value=httpx.Response(
            200,
            json=_search_response(
                [
                    {"setid": "bbbb-1", "spl_version": "1", "title": "GENERIC DRUG A", "published_date": "Jan 1, 2020"},
                    {"setid": "bbbb-2", "spl_version": "1", "title": "GENERIC DRUG B", "published_date": "Jan 1, 2021"},
                ]
            ),
        )
    )
    respx.get(f"{BASE_URL}/spls/bbbb-1.xml").mock(
        return_value=httpx.Response(
            200,
            text=_spl_xml("bbbb-1", "GENERIC DRUG A", include_interactions_section=False, contraindications_text="None known."),
        )
    )
    respx.get(f"{BASE_URL}/spls/bbbb-2.xml").mock(
        return_value=httpx.Response(
            200,
            # Section present in XML but only a placeholder "." — must be treated as NOT found.
            text=_spl_xml("bbbb-2", "GENERIC DRUG B", interactions_text=None, contraindications_text=None),
        )
    )

    result = dailymed_service.get_drug_label_info(rxcui="999", drug_name="genericdrug")

    assert result.evidence_status == "label_found_no_interaction_section"
    assert result.candidates_checked == 2
    assert result.total_candidates_found == 2
    assert result.sections["drug_interactions"].found is False
    assert any("Checked 2 of 2" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Tests: placeholder "." content is correctly rejected as non-meaningful
# ---------------------------------------------------------------------------

@respx.mock
def test_placeholder_dot_is_not_meaningful_content():
    respx.get(f"{BASE_URL}/spls.json", params={"rxcui": "555", "pagesize": "10"}).mock(
        return_value=httpx.Response(
            200,
            json=_search_response(
                [{"setid": "cccc-1", "spl_version": "1", "title": "SOME DRUG", "published_date": "Jan 1, 2020"}]
            ),
        )
    )
    respx.get(f"{BASE_URL}/spls/cccc-1.xml").mock(
        return_value=httpx.Response(200, text=_spl_xml("cccc-1", "SOME DRUG", interactions_text=None))
    )

    result = dailymed_service.get_drug_label_info(rxcui="555")

    assert result.sections["drug_interactions"].found is False
    assert result.sections["drug_interactions"].text == ""
    assert result.evidence_status == "label_found_no_interaction_section"


# ---------------------------------------------------------------------------
# Tests: no label found at all
# ---------------------------------------------------------------------------

@respx.mock
def test_get_label_info_no_label_found():
    respx.get(f"{BASE_URL}/spls.json", params={"rxcui": "0", "pagesize": "10"}).mock(
        return_value=httpx.Response(200, json=_search_response([]))
    )
    respx.get(f"{BASE_URL}/spls.json", params={"drug_name": "notarealdrug", "pagesize": "10"}).mock(
        return_value=httpx.Response(200, json=_search_response([]))
    )

    result = dailymed_service.get_drug_label_info(rxcui="0", drug_name="notarealdrug")

    assert result.evidence_status == "no_label_found"
    assert result.setid is None
    assert len(result.warnings) == 1


# ---------------------------------------------------------------------------
# Tests: falls back to drug_name search when rxcui search yields nothing
# ---------------------------------------------------------------------------

@respx.mock
def test_falls_back_to_drug_name_search():
    respx.get(f"{BASE_URL}/spls.json", params={"rxcui": "123", "pagesize": "10"}).mock(
        return_value=httpx.Response(200, json=_search_response([]))
    )
    respx.get(f"{BASE_URL}/spls.json", params={"drug_name": "ibuprofen", "pagesize": "10"}).mock(
        return_value=httpx.Response(
            200,
            json=_search_response(
                [{"setid": "dddd-1", "spl_version": "2", "title": "IBUPROFEN TABLET", "published_date": "Jan 1, 2022"}]
            ),
        )
    )
    respx.get(f"{BASE_URL}/spls/dddd-1.xml").mock(
        return_value=httpx.Response(200, text=_spl_xml("dddd-1", "IBUPROFEN TABLET"))
    )

    result = dailymed_service.get_drug_label_info(rxcui="123", drug_name="ibuprofen")

    assert result.evidence_status == "interaction_evidence_found"
    assert result.setid == "dddd-1"


# ---------------------------------------------------------------------------
# Tests: no identifiers supplied at all
# ---------------------------------------------------------------------------

def test_no_identifiers_supplied():
    result = dailymed_service.get_drug_label_info()
    assert result.evidence_status == "error"
    assert result.error is not None


# ---------------------------------------------------------------------------
# Tests: API failures distinguished from "no label"
# ---------------------------------------------------------------------------

@respx.mock
def test_search_5xx_returns_error_not_no_label_found():
    respx.get(f"{BASE_URL}/spls.json", params={"rxcui": "11289", "pagesize": "10"}).mock(
        return_value=httpx.Response(503, text="Service Unavailable")
    )

    result = dailymed_service.get_drug_label_info(rxcui="11289")

    assert result.evidence_status == "error"
    assert result.error is not None


@respx.mock
def test_label_fetch_timeout_returns_error():
    respx.get(f"{BASE_URL}/spls.json", params={"rxcui": "11289", "pagesize": "10"}).mock(
        return_value=httpx.Response(
            200,
            json=_search_response(
                [{"setid": "eeee-1", "spl_version": "1", "title": "SOME DRUG", "published_date": "Jan 1, 2020"}]
            ),
        )
    )
    respx.get(f"{BASE_URL}/spls/eeee-1.xml").mock(side_effect=httpx.TimeoutException("timed out"))

    result = dailymed_service.get_drug_label_info(rxcui="11289")

    assert result.evidence_status == "error"
    assert "timed out" in result.error.lower() or "timeout" in result.error.lower()


@respx.mock
def test_malformed_xml_returns_error():
    respx.get(f"{BASE_URL}/spls.json", params={"rxcui": "11289", "pagesize": "10"}).mock(
        return_value=httpx.Response(
            200,
            json=_search_response(
                [{"setid": "ffff-1", "spl_version": "1", "title": "SOME DRUG", "published_date": "Jan 1, 2020"}]
            ),
        )
    )
    respx.get(f"{BASE_URL}/spls/ffff-1.xml").mock(
        return_value=httpx.Response(200, text="<this is not valid xml")
    )

    result = dailymed_service.get_drug_label_info(rxcui="11289")

    assert result.evidence_status == "error"
    assert result.error is not None


@respx.mock
def test_search_404_treated_as_no_results_not_error():
    """DailyMed sometimes returns 404 for a search with zero matches rather
    than an empty 200 body — must be handled the same as 'no results',
    not raised as an API error."""
    respx.get(f"{BASE_URL}/spls.json", params={"rxcui": "77", "pagesize": "10"}).mock(
        return_value=httpx.Response(404, text="Not Found")
    )
    respx.get(f"{BASE_URL}/spls.json", params={"drug_name": "obscuredrug", "pagesize": "10"}).mock(
        return_value=httpx.Response(404, text="Not Found")
    )

    result = dailymed_service.get_drug_label_info(rxcui="77", drug_name="obscuredrug")

    assert result.evidence_status == "no_label_found"


# ---------------------------------------------------------------------------
# Tests: XML parsing helper functions directly
# ---------------------------------------------------------------------------

def test_is_meaningful_text_rejects_placeholders():
    assert dailymed_service._is_meaningful_text(".") is False
    assert dailymed_service._is_meaningful_text("") is False
    assert dailymed_service._is_meaningful_text("   ") is False
    assert dailymed_service._is_meaningful_text("N/A") is False
    assert dailymed_service._is_meaningful_text("None") is False


def test_is_meaningful_text_accepts_real_content():
    assert dailymed_service._is_meaningful_text("May increase bleeding risk.") is True


def test_parse_label_xml_extracts_metadata_and_sections():
    xml_text = _spl_xml("gggg-1", "TEST DRUG TABLET", version="5", effective_time="20230601")
    parsed = dailymed_service._parse_label_xml(xml_text)

    assert parsed["setid"] == "gggg-1"
    assert parsed["spl_version"] == "5"
    assert parsed["effective_time"] == "20230601"
    assert parsed["title"] == "TEST DRUG TABLET"
    assert parsed["sections"]["drug_interactions"].found is True
    assert parsed["sections"]["boxed_warning"].found is False  # not included in fixture


def test_parse_label_xml_extracts_manufacturer_when_present():
    xml_with_author = """<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="urn:hl7-org:v3">
  <id root="hhhh-doc"/>
  <setId root="hhhh-1"/>
  <versionNumber value="1"/>
  <title>TEST DRUG</title>
  <effectiveTime value="20230601"/>
  <author>
    <assignedEntity>
      <representedOrganization>
        <name>Example Pharma Inc.</name>
      </representedOrganization>
    </assignedEntity>
  </author>
  <component><structuredBody></structuredBody></component>
</document>"""
    parsed = dailymed_service._parse_label_xml(xml_with_author)
    assert parsed["manufacturer"] == "Example Pharma Inc."


def test_parse_label_xml_manufacturer_none_when_absent():
    xml_text = _spl_xml("iiii-1", "TEST DRUG TABLET")  # fixture has no <author>
    parsed = dailymed_service._parse_label_xml(xml_text)
    assert parsed["manufacturer"] is None


def test_parse_label_xml_raises_on_malformed_xml():
    from app.exceptions import ExternalAPIBadResponseError

    with pytest.raises(ExternalAPIBadResponseError):
        dailymed_service._parse_label_xml("<not><valid</xml>")
