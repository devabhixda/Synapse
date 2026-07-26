from synapse.inference.assertions import normalize_kind, normalize_predicate


def test_assertion_kinds_are_limited_to_graph_taxonomy() -> None:
    assert normalize_kind("product") == "Product"
    assert normalize_kind("invented type") == "Concept"


def test_assertion_predicates_map_to_stable_ontology() -> None:
    assert normalize_predicate("request_api_access") == "requests"
    assert normalize_predicate("fix bug") == "resolves"
    assert normalize_predicate("develop-dashboard") == "implements"
    assert normalize_predicate("unexpected relation") == "related_to"
