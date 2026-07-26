from synapse.config import Settings
from synapse.enrichment.crustdata import (
    CrustdataEnricher,
    parse_enrich_response,
    parse_identify_response,
)


async def test_crustdata_fixture_enriches_only_known_domain() -> None:
    enricher = CrustdataEnricher(Settings(crustdata_api_key=None))

    result = await enricher.enrich_company("Crustdata", "crustdata.com")
    missing = await enricher.enrich_company("Unknown", "unknown.invalid")

    assert result is not None
    assert result.provider == "crustdata-fixture"
    assert result.payload["basic_info"]["primary_domain"] == "crustdata.com"
    assert missing is None


def test_crustdata_response_parsers_require_domain_match() -> None:
    identify = [
        {
            "matches": [
                {
                    "confidence_score": 0.91,
                    "company_data": {
                        "crustdata_company_id": 42,
                        "basic_info": {"primary_domain": "crustdata.com"},
                    },
                }
            ]
        }
    ]
    enrich = [{"matches": [{"company_data": {"headcount": {"total": 12}}}]}]

    assert parse_identify_response(identify, "crustdata.com") == (42, 0.91)
    assert parse_identify_response(identify, "other.com") == (None, 0.0)
    assert parse_enrich_response(enrich) == {"headcount": {"total": 12}}