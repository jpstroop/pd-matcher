"""Round-trip test for the shipped copyright_rules.yaml."""

from msgspec import convert
from msgspec import to_builtins

from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.copyright import default_ruleset


def test_shipped_ruleset_roundtrips_through_msgspec() -> None:
    """The shipped ruleset survives a builtins -> Struct round-trip."""
    original = default_ruleset()
    again = convert(to_builtins(original), type=CopyrightRuleSet)
    assert again == original
    assert again.version == "1.0.0"
    assert len(again.rules) == len(original.rules)


def test_shipped_ruleset_predicate_names_are_registered() -> None:
    """Every predicate name used in the shipped YAML is registered."""
    from pd_matcher.copyright.rules import registered_predicate_names

    known = set(registered_predicate_names())
    for rule in default_ruleset().rules:
        for call in rule.when:
            assert call.predicate in known, (
                f"rule {rule.name!r} references unknown predicate {call.predicate!r}"
            )
