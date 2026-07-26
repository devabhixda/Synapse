from fastapi.testclient import TestClient

from synapse.api.main import app


def test_openapi_exposes_health() -> None:
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
    assert "/api/health" in schema["paths"]
    assert "/api/system2/consolidate" in schema["paths"]
    assert "/api/system2/runs" in schema["paths"]
    assert "/api/system2/graph" in schema["paths"]
    assert "/api/recall" in schema["paths"]
    assert "/api/procedural/detect" in schema["paths"]
    assert "/api/procedural/patterns" in schema["paths"]
    assert "/api/procedural/skills" in schema["paths"]