"""Tests for :mod:`pd_matcher.copyright.status`."""

from pd_matcher.copyright import default_ruleset
from pd_matcher.copyright.status import CopyrightStatus


def test_every_status_member_is_produced_by_a_rule_or_short_circuit() -> None:
    """Every enum member must be reachable from the shipped ruleset.

    The moving-wall short-circuit produces ``PD_BY_AGE_PRE_95_YEARS``;
    the fallback path produces ``UNKNOWN_NO_RULE_MATCHED``; every other
    member must appear as the ``then`` of at least one rule, or as the
    ``on_coverage_fail`` short-circuit on a coverage-aware rule.
    """
    produced: set[str] = set()
    for rule in default_ruleset().rules:
        produced.add(rule.then)
        if rule.on_coverage_fail is not None:
            produced.add(rule.on_coverage_fail)
    produced.add(CopyrightStatus.PD_BY_AGE_PRE_95_YEARS.name)
    produced.add(CopyrightStatus.UNKNOWN_NO_RULE_MATCHED.name)
    missing = {s.name for s in CopyrightStatus} - produced
    assert missing == set(), f"unreachable enum members: {sorted(missing)}"


def test_str_value_equals_member_name() -> None:
    """Members are ``StrEnum`` with values equal to their names."""
    for member in CopyrightStatus:
        assert str(member) == member.name
