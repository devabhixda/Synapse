import pytest
from pydantic import ValidationError

from synapse.config import DEFAULT_ENTITY_KINDS, Settings
from synapse.inference.assertions import normalize_kind, normalize_predicate, system_prompt

PERSONAL_KINDS = ["Person", "Place", "Event", "Organization", "Preference", "Task", "Concept"]


def test_assertion_kinds_are_limited_to_graph_taxonomy() -> None:
    assert normalize_kind("product") == "Product"
    assert normalize_kind("invented type") == "Concept"


def test_configured_kinds_replace_the_workplace_ones() -> None:
    assert normalize_kind("place", PERSONAL_KINDS) == "Place"
    assert normalize_kind(" ORGANIZATION ", PERSONAL_KINDS) == "Organization"
    assert normalize_kind("Repository", PERSONAL_KINDS) == "Concept"


def test_a_kind_keeps_its_configured_spelling() -> None:
    assert normalize_kind("tv  show", ["TV Show"]) == "TV Show"


def test_settings_default_to_the_workplace_kinds() -> None:
    assert Settings().entity_kinds == list(DEFAULT_ENTITY_KINDS)


def test_configured_kinds_always_include_the_fallback_kind() -> None:
    settings = Settings(entity_kinds=["Person", "Place", "Person", " Event "])

    assert settings.entity_kinds == ["Person", "Place", "Event", "Concept"]
    assert Settings(entity_kinds=["person", "PERSON", "concept"]).entity_kinds == [
        "person",
        "Concept",
    ]
    with pytest.raises(ValidationError):
        Settings(entity_kinds=["Person", " "])


def test_the_prompt_names_the_configured_kinds() -> None:
    prompt = system_prompt(PERSONAL_KINDS)

    assert (
        "Allowed kinds:\nPerson, Place, Event, Organization, Preference, Task, Concept."
        in prompt
    )
    assert "Repository" not in prompt
    assert "Repository" in system_prompt()


def test_assertion_predicates_map_to_stable_ontology() -> None:
    assert normalize_predicate("request_api_access") == "requests"
    assert normalize_predicate("fix bug") == "resolves"
    assert normalize_predicate("develop-dashboard") == "implements"
    assert normalize_predicate("unexpected relation") == "related_to"
