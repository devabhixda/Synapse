from synapse.procedural.service import classify_recall_family


def test_recall_queries_are_grouped_into_stable_workflow_families() -> None:
    assert (
        classify_recall_family("What industry is Crustdata in and how large is it?")
        == "company_research"
    )
    assert (
        classify_recall_family("Tell me Crustdata company size and sector")
        == "company_research"
    )
    assert (
        classify_recall_family("What database does Atlas API use?")
        == "architecture_context"
    )
    assert (
        classify_recall_family("Give me Titan migration status and blockers")
        == "project_status_briefing"
    )