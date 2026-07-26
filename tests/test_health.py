from fastapi.testclient import TestClient

from synapse.api.main import app


def test_openapi_exposes_health() -> None:
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
    assert "/api/health" in schema["paths"]