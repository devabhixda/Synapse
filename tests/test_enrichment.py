from synapse.config import Settings
from synapse.enrichment.crustdata import (
    CrustdataEnricher,
    parse_enrich_response,
    parse_identify_response,
)


async def test_crustdata_without_api_key_does_not_invent_enrichment() -> None:
    enricher = CrustdataEnricher(Settings(crustdata_api_key=None))

    result = await enricher.enrich_company("Orbit Forge", "orbit.invalid")

    assert result is None


def test_crustdata_response_parsers_require_domain_match() -> None:
    identify = [
        {
            "matches": [
                {
                    "confidence_score": 0.91,
                    "company_data": {
                        "crustdata_company_id": 42,
                        "basic_info": {"primary_domain": "orbit.example"},
                    },
                }
            ]
        }
    ]
    enrich = [{"matches": [{"company_data": {"headcount": {"total": 12}}}]}]

    assert parse_identify_response(identify, "orbit.example") == (42, 0.91)
    assert parse_identify_response(identify, "other.com") == (None, 0.0)
    assert parse_enrich_response(enrich) == {"headcount": {"total": 12}}