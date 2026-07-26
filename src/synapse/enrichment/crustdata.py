from typing import Any, Protocol

import httpx

from synapse.config import Settings, get_settings
from synapse.enrichment.models import EnrichmentResult


class CompanyEnricher(Protocol):
    async def enrich_company(self, name: str, domain: str | None) -> EnrichmentResult | None: ...


class CrustdataEnricher:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def enrich_company(self, name: str, domain: str | None) -> EnrichmentResult | None:
        if not domain:
            return None
        if not self.settings.crustdata_api_key:
            return fixture_enrichment(name, domain)

        headers = {
            "Authorization": f"Bearer {self.settings.crustdata_api_key}",
            "Content-Type": "application/json",
            "x-api-version": self.settings.crustdata_api_version,
        }
        async with httpx.AsyncClient(
            base_url=self.settings.crustdata_base_url, headers=headers, timeout=20
        ) as client:
            identify_response = await client.post(
                "/company/identify", json={"domains": [domain], "exact_match": True}
            )
            identify_response.raise_for_status()
            company_id, confidence = parse_identify_response(identify_response.json(), domain)
            if company_id is None:
                return None

            enrich_response = await client.post(
                "/company/enrich",
                json={
                    "crustdata_company_ids": [company_id],
                    "fields": ["basic_info", "headcount", "funding"],
                },
            )
            enrich_response.raise_for_status()
            payload = parse_enrich_response(enrich_response.json())
            if payload is None:
                return None
            return EnrichmentResult(
                provider="crustdata",
                field_group="company_profile",
                payload=payload,
                source_url=f"{self.settings.crustdata_base_url}/company/enrich",
                confidence=confidence,
            )


def parse_identify_response(payload: object, domain: str) -> tuple[int | None, float]:
    if not isinstance(payload, list) or not payload:
        return None, 0.0
    first = payload[0]
    if not isinstance(first, dict):
        return None, 0.0
    matches = first.get("matches")
    if not isinstance(matches, list):
        return None, 0.0

    valid_matches: list[tuple[float, int]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        company_data = match.get("company_data")
        if not isinstance(company_data, dict):
            continue
        basic_info = company_data.get("basic_info")
        if not isinstance(basic_info, dict):
            continue
        primary_domain = str(basic_info.get("primary_domain") or "").casefold()
        company_id = company_data.get("crustdata_company_id") or basic_info.get(
            "crustdata_company_id"
        )
        if primary_domain == domain.casefold() and isinstance(company_id, int):
            confidence = float(match.get("confidence_score") or 0)
            valid_matches.append((confidence, company_id))
    if not valid_matches:
        return None, 0.0
    confidence, company_id = max(valid_matches)
    return company_id, min(max(confidence, 0.0), 1.0)


def parse_enrich_response(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    if not isinstance(first, dict):
        return None
    matches = first.get("matches")
    if not isinstance(matches, list) or not matches:
        return None
    match = matches[0]
    if not isinstance(match, dict):
        return None
    company_data = match.get("company_data")
    return company_data if isinstance(company_data, dict) else None


def fixture_enrichment(name: str, domain: str) -> EnrichmentResult | None:
    if domain.casefold() != "crustdata.com":
        return None
    return EnrichmentResult(
        provider="crustdata-fixture",
        field_group="company_profile",
        payload={
            "basic_info": {
                "name": name,
                "primary_domain": "crustdata.com",
                "description": "Real-time company and people data APIs for AI agents.",
                "industries": ["B2B data", "Artificial intelligence"],
            },
            "coverage": {
                "people_profiles": "1B+",
                "company_profiles": "60M+",
            },
        },
        source_url="https://crustdata.com",
        confidence=1.0,
    )