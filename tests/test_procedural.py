from synapse.procedural.service import classify_recall_family, is_procedural_evidence


def test_recall_queries_are_grouped_into_stable_workflow_families() -> None:
    assert (
        classify_recall_family("What industry is Acme Robotics in and how large is it?")
        == "company_research"
    )
    assert (
        classify_recall_family("Tell me Orbit Forge company size and sector")
        == "company_research"
    )
    assert (
        classify_recall_family("What database does the billing service use?")
        == "architecture_context"
    )
    assert (
        classify_recall_family("Give me the launch status and blockers")
        == "project_status_briefing"
    )


def test_only_successful_recalls_are_procedural_evidence() -> None:
    assert is_procedural_evidence("episodic")
    assert is_procedural_evidence("semantic_fallback")
    assert not is_procedural_evidence("miss")