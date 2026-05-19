"""Tests for :mod:`pd_matcher.copyright.rules`."""

from pytest import raises

from pd_matcher.config.schemas import CopyrightRule
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import PredicateCall
from pd_matcher.copyright.rules import RuleEvaluationError
from pd_matcher.copyright.rules import assess
from pd_matcher.copyright.rules import registered_predicate_names
from pd_matcher.copyright.status import CopyrightStatus
from tests.unit.copyright.conftest import make_facts


def _ruleset_for(rule: CopyrightRule) -> CopyrightRuleSet:
    """Wrap one ``CopyrightRule`` in a minimal ``CopyrightRuleSet``."""
    return CopyrightRuleSet(version="test", rules=[rule])


def test_moving_wall_short_circuit_returns_pd_by_age() -> None:
    """``pub_year < as_of_year - 95`` should not consult any rule."""
    sentinel_rule = CopyrightRule(
        name="sentinel",
        when=[PredicateCall(predicate="country_is_us")],
        then="UNKNOWN_NO_RULE_MATCHED",
        explanation="should not be reached",
    )
    facts = make_facts(pub_year=1900, pub_country_code="nyu")
    result = assess(facts, _ruleset_for(sentinel_rule))
    assert result.status is CopyrightStatus.PD_BY_AGE_PRE_95_YEARS
    assert result.matched_rule_name == "moving_wall_short_circuit"
    assert "Published in 1900" in result.explanation
    assert result.assumptions == ()


def test_fallback_status_when_no_rule_matches() -> None:
    """An empty ruleset returns ``UNKNOWN_NO_RULE_MATCHED``."""
    facts = make_facts(pub_year=1985, pub_country_code="nyu")
    ruleset = CopyrightRuleSet(version="empty", rules=[])
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.UNKNOWN_NO_RULE_MATCHED
    assert result.matched_rule_name is None
    assert result.assumptions == ()
    assert "No rule matched" in result.explanation


def test_unknown_predicate_raises() -> None:
    """A rule referencing an unknown predicate must fail loudly."""
    rule = CopyrightRule(
        name="bad",
        when=[PredicateCall(predicate="nonexistent")],
        then="UNKNOWN_NO_RULE_MATCHED",
        explanation="x",
    )
    facts = make_facts(pub_year=1985, pub_country_code="nyu")
    with raises(RuleEvaluationError, match="unknown predicate"):
        assess(facts, _ruleset_for(rule))


def test_unknown_status_raises() -> None:
    """A rule pointing at an unknown CopyrightStatus must fail loudly."""
    rule = CopyrightRule(
        name="bad_status",
        when=[],
        then="NOT_A_REAL_STATUS",
        explanation="x",
    )
    facts = make_facts(pub_year=1985, pub_country_code="nyu")
    with raises(RuleEvaluationError, match="unknown CopyrightStatus"):
        assess(facts, _ruleset_for(rule))


def test_zero_arg_predicate_rejects_extra_args() -> None:
    """Zero-arg predicates must reject any positional arguments."""
    rule = CopyrightRule(
        name="bad_arity",
        when=[PredicateCall(predicate="country_is_us", args=(1,))],
        then="UNKNOWN_NO_RULE_MATCHED",
        explanation="x",
    )
    facts = make_facts(pub_year=1985, pub_country_code="nyu")
    with raises(RuleEvaluationError, match="expects no args"):
        assess(facts, _ruleset_for(rule))


def test_int_int_predicate_rejects_wrong_arity() -> None:
    """``published_between`` must receive exactly two arguments."""
    rule = CopyrightRule(
        name="bad_arity",
        when=[PredicateCall(predicate="published_between", args=(1931,))],
        then="UNKNOWN_NO_RULE_MATCHED",
        explanation="x",
    )
    facts = make_facts(pub_year=1985, pub_country_code="nyu")
    with raises(RuleEvaluationError, match="expects 2 args"):
        assess(facts, _ruleset_for(rule))


def test_int_predicate_rejects_wrong_arity() -> None:
    """Single-int predicates must receive exactly one argument."""
    rule = CopyrightRule(
        name="bad_arity",
        when=[PredicateCall(predicate="published_before", args=(1, 2))],
        then="UNKNOWN_NO_RULE_MATCHED",
        explanation="x",
    )
    facts = make_facts(pub_year=1985, pub_country_code="nyu")
    with raises(RuleEvaluationError, match="expects 1 arg"):
        assess(facts, _ruleset_for(rule))


def test_float_predicate_rejects_wrong_arity() -> None:
    """``match_confidence_at_least`` must receive exactly one argument."""
    rule = CopyrightRule(
        name="bad_arity",
        when=[PredicateCall(predicate="match_confidence_at_least", args=())],
        then="UNKNOWN_NO_RULE_MATCHED",
        explanation="x",
    )
    facts = make_facts(pub_year=1985, pub_country_code="nyu")
    with raises(RuleEvaluationError, match="expects 1 arg"):
        assess(facts, _ruleset_for(rule))


def test_inference_predicate_rejects_args() -> None:
    """Inference functions take no positional arguments."""
    rule = CopyrightRule(
        name="bad_arity",
        when=[PredicateCall(predicate="has_us_notice", args=(1,))],
        then="UNKNOWN_NO_RULE_MATCHED",
        explanation="x",
    )
    facts = make_facts(pub_year=1985, pub_country_code="nyu", was_registered=True)
    with raises(RuleEvaluationError, match="expects no args"):
        assess(facts, _ruleset_for(rule))


def test_negate_flips_predicate_value() -> None:
    """A negated predicate that would return False matches; the rule fires."""
    rule = CopyrightRule(
        name="not_renewed",
        when=[
            PredicateCall(predicate="country_is_us"),
            PredicateCall(predicate="was_renewed", negate=True),
        ],
        then="PD_REGISTERED_NOT_RENEWED",
        explanation="not renewed",
    )
    facts = make_facts(
        pub_year=1950,
        pub_country_code="nyu",
        was_registered=True,
        was_renewed=False,
    )
    result = assess(facts, _ruleset_for(rule))
    assert result.status is CopyrightStatus.PD_REGISTERED_NOT_RENEWED


def test_enable_assumptions_false_blocks_inference_rules() -> None:
    """When assumptions are disabled, inference predicates cannot satisfy a rule."""
    rule = CopyrightRule(
        name="needs_notice",
        when=[
            PredicateCall(predicate="country_is_us"),
            PredicateCall(predicate="has_us_notice"),
        ],
        then="IN_COPYRIGHT_REGISTERED_AND_RENEWED",
        explanation="needs notice",
    )
    facts = make_facts(
        pub_year=1950,
        pub_country_code="nyu",
        was_registered=True,
    )
    result = assess(facts, _ruleset_for(rule), enable_assumptions=False)
    assert result.status is CopyrightStatus.UNKNOWN_NO_RULE_MATCHED


def test_static_and_dynamic_assumptions_concatenated() -> None:
    """Static rule-level assumptions and dynamic inference assumptions both surface."""
    rule = CopyrightRule(
        name="combo",
        when=[
            PredicateCall(predicate="country_is_us"),
            PredicateCall(predicate="has_us_notice"),
        ],
        then="PD_REGISTERED_NOT_RENEWED",
        explanation="combo",
        assumptions=["Static assumption"],
    )
    facts = make_facts(
        pub_year=1950,
        pub_country_code="nyu",
        was_registered=True,
    )
    result = assess(facts, _ruleset_for(rule))
    assert result.assumptions[0] == "Static assumption"
    assert any("registration" in a.lower() for a in result.assumptions[1:])


def test_negated_inference_does_not_contribute_assumption() -> None:
    """When negation flips an inference to True, no assumption is added.

    The negation case is "we observed that this was NOT a US-gov work";
    that is a direct observation, not a documented leap. We therefore
    refuse to surface the inference's positive-direction assumption.
    """
    rule = CopyrightRule(
        name="not_us_gov",
        when=[
            PredicateCall(predicate="country_is_us"),
            PredicateCall(predicate="is_us_government_work", negate=True),
        ],
        then="IN_COPYRIGHT_US_PUB_POST_1989",
        explanation="not us-gov",
    )
    facts = make_facts(
        pub_year=2010,
        pub_country_code="nyu",
        publisher_text="Penguin Random House",
    )
    result = assess(facts, _ruleset_for(rule))
    assert result.status is CopyrightStatus.IN_COPYRIGHT_US_PUB_POST_1989
    assert result.assumptions == ()


def test_float_predicate_fires_when_threshold_met() -> None:
    """``match_confidence_at_least`` evaluated through a rule returns the right verdict."""
    rule = CopyrightRule(
        name="needs_confidence",
        when=[
            PredicateCall(
                predicate="match_confidence_at_least",
                args=(0.5,),
            ),
        ],
        then="IN_COPYRIGHT_US_PUB_POST_1989",
        explanation="confidence",
    )
    facts = make_facts(
        pub_year=2010,
        pub_country_code="nyu",
        match_confidence=0.9,
    )
    result = assess(facts, _ruleset_for(rule))
    assert result.status is CopyrightStatus.IN_COPYRIGHT_US_PUB_POST_1989


def test_registered_predicate_names_includes_every_registered_predicate() -> None:
    """The registry introspection helper returns the expected name set."""
    names = set(registered_predicate_names())
    expected = {
        "in_pd_by_age",
        "country_is_us",
        "country_is_foreign",
        "country_no_treaty",
        "country_delayed_uraa",
        "was_registered",
        "was_renewed",
        "published_between",
        "published_before",
        "published_on_or_after",
        "match_confidence_at_least",
        "has_us_notice",
        "is_us_government_work",
        "foreign_in_pd_home_country_1996",
    }
    assert names == expected


# -----------------------------------------------------------------------
# End-to-end rule firing tests: one per shipped YAML rule.               #
# Each test constructs Facts that fires exactly one rule.                #
# -----------------------------------------------------------------------


def test_shipped_us_government_work(ruleset: CopyrightRuleSet) -> None:
    facts = make_facts(
        pub_year=1955,
        pub_country_code="dcu",
        publisher_text="U.S. Government Printing Office",
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.PD_US_GOVERNMENT_WORK


def test_shipped_us_pub_1931_1977_not_registered(ruleset: CopyrightRuleSet) -> None:
    facts = make_facts(
        pub_year=1950,
        pub_country_code="nyu",
        was_registered=False,
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.PD_US_PUB_NO_NOTICE_1931_1977


def test_shipped_us_pub_1931_1963_registered_not_renewed(
    ruleset: CopyrightRuleSet,
) -> None:
    facts = make_facts(
        pub_year=1950,
        pub_country_code="nyu",
        was_registered=True,
        was_renewed=False,
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.PD_REGISTERED_NOT_RENEWED


def test_shipped_us_pub_1931_1963_registered_and_renewed(
    ruleset: CopyrightRuleSet,
) -> None:
    facts = make_facts(
        pub_year=1950,
        pub_country_code="nyu",
        was_registered=True,
        was_renewed=True,
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.IN_COPYRIGHT_REGISTERED_AND_RENEWED


def test_shipped_us_pub_1964_1977_with_notice(ruleset: CopyrightRuleSet) -> None:
    facts = make_facts(
        pub_year=1970,
        pub_country_code="nyu",
        was_registered=True,
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.IN_COPYRIGHT_1964_1977_WITH_NOTICE


def test_shipped_us_pub_1978_1989_no_registration(
    ruleset: CopyrightRuleSet,
) -> None:
    facts = make_facts(
        pub_year=1985,
        pub_country_code="nyu",
        was_registered=False,
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.PD_US_PUB_NO_REGISTRATION_1978_1989


def test_shipped_us_pub_1978_1989_registered(ruleset: CopyrightRuleSet) -> None:
    facts = make_facts(
        pub_year=1985,
        pub_country_code="nyu",
        was_registered=True,
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.IN_COPYRIGHT_1978_1989_CURED


def test_shipped_us_pub_post_1989(ruleset: CopyrightRuleSet) -> None:
    facts = make_facts(
        pub_year=2010,
        pub_country_code="nyu",
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.IN_COPYRIGHT_US_PUB_POST_1989


def test_shipped_foreign_no_treaty_country(ruleset: CopyrightRuleSet) -> None:
    facts = make_facts(pub_year=1970, pub_country_code="er")
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.PD_FOREIGN_NO_TREATY_COUNTRY


def test_shipped_foreign_pre_1923_pd_home_country_1996(
    ruleset: CopyrightRuleSet,
) -> None:
    """Pinned ``as_of_year=2017`` to land between the moving wall and the 1923 cutoff."""
    facts = make_facts(pub_year=1922, pub_country_code="fr", as_of_year=2017)
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.PD_FOREIGN_IN_HOME_COUNTRY_PD_1996


def test_shipped_foreign_uraa_restored(ruleset: CopyrightRuleSet) -> None:
    facts = make_facts(pub_year=1950, pub_country_code="fr")
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.IN_COPYRIGHT_FOREIGN_URAA_RESTORED


def test_shipped_foreign_1978_2002_floor(ruleset: CopyrightRuleSet) -> None:
    facts = make_facts(pub_year=1985, pub_country_code="fr")
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.IN_COPYRIGHT_PRE_1978_PUBLISHED_1978_2002_FLOOR


def test_shipped_foreign_post_1989(ruleset: CopyrightRuleSet) -> None:
    facts = make_facts(pub_year=2010, pub_country_code="fr")
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.IN_COPYRIGHT_FOREIGN_POST_1989


def test_shipped_foreign_delayed_uraa_unknown(ruleset: CopyrightRuleSet) -> None:
    facts = make_facts(pub_year=1985, pub_country_code="af")
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.UNKNOWN_INSUFFICIENT_DATA


# Negative tests: exercise the False branch of each non-trivial rule by
# flipping exactly one predicate's input.


def test_us_pub_registered_not_renewed_negative_year_outside_window(
    ruleset: CopyrightRuleSet,
) -> None:
    """A year outside 1931-1963 prevents the not-renewed rule from firing."""
    facts = make_facts(
        pub_year=1970,
        pub_country_code="nyu",
        was_registered=True,
        was_renewed=False,
    )
    result = assess(facts, ruleset)
    assert result.status is not CopyrightStatus.PD_REGISTERED_NOT_RENEWED


def test_foreign_no_treaty_negative_other_country(
    ruleset: CopyrightRuleSet,
) -> None:
    """A country with treaty relations cannot fire the no-treaty rule."""
    facts = make_facts(pub_year=1985, pub_country_code="fr")
    result = assess(facts, ruleset)
    assert result.status is not CopyrightStatus.PD_FOREIGN_NO_TREATY_COUNTRY


# -----------------------------------------------------------------------
# Foreign-registered routing: a foreign-authored work that registered    #
# with the US Copyright Office must follow Category 2 (registration     #
# gates the formality-failure logic), not Category 3 URAA.              #
# -----------------------------------------------------------------------


def test_foreign_registered_and_renewed_follows_category_2(
    ruleset: CopyrightRuleSet,
) -> None:
    """Foreign work that registered AND renewed -> IN_COPYRIGHT_REGISTERED_AND_RENEWED.

    URAA restores works that failed the US formalities; a surviving CCE
    registration *is* the US formality, so the work follows Category 2.
    """
    facts = make_facts(
        pub_year=1950,
        pub_country_code="fr",
        was_registered=True,
        was_renewed=True,
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.IN_COPYRIGHT_REGISTERED_AND_RENEWED


def test_foreign_registered_not_renewed_follows_category_2(
    ruleset: CopyrightRuleSet,
) -> None:
    """Foreign work registered but never renewed -> PD_REGISTERED_NOT_RENEWED."""
    facts = make_facts(
        pub_year=1950,
        pub_country_code="fr",
        was_registered=True,
        was_renewed=False,
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.PD_REGISTERED_NOT_RENEWED


def test_foreign_registered_1964_1977_follows_category_2(
    ruleset: CopyrightRuleSet,
) -> None:
    """Foreign work registered 1964-1977 -> IN_COPYRIGHT_1964_1977_WITH_NOTICE."""
    facts = make_facts(
        pub_year=1970,
        pub_country_code="fr",
        was_registered=True,
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.IN_COPYRIGHT_1964_1977_WITH_NOTICE


def test_foreign_registered_1978_1989_cured_follows_category_2(
    ruleset: CopyrightRuleSet,
) -> None:
    """Foreign work registered in 1978-Feb 1989 -> IN_COPYRIGHT_1978_1989_CURED."""
    facts = make_facts(
        pub_year=1985,
        pub_country_code="fr",
        was_registered=True,
    )
    result = assess(facts, ruleset)
    assert result.status is CopyrightStatus.IN_COPYRIGHT_1978_1989_CURED


def test_foreign_registered_does_not_fire_uraa_restored(
    ruleset: CopyrightRuleSet,
) -> None:
    """A foreign-registered 1931-1977 work must NOT yield URAA_RESTORED."""
    facts = make_facts(
        pub_year=1950,
        pub_country_code="fr",
        was_registered=True,
        was_renewed=True,
    )
    result = assess(facts, ruleset)
    assert result.status is not CopyrightStatus.IN_COPYRIGHT_FOREIGN_URAA_RESTORED


def test_foreign_registered_does_not_fire_1978_2002_floor(
    ruleset: CopyrightRuleSet,
) -> None:
    """A foreign-registered 1978-2002 work must NOT yield the 2047 floor.

    A registered work in 1978-1989 gets the registration cure
    (Category 2). A registered work 1990-2002 falls through Category 3
    URAA rules (all negated on registration) and lands on
    ``UNKNOWN_NO_RULE_MATCHED``, but never on the floor leaf.
    """
    facts = make_facts(
        pub_year=1985,
        pub_country_code="fr",
        was_registered=True,
    )
    result = assess(facts, ruleset)
    assert result.status is not CopyrightStatus.IN_COPYRIGHT_PRE_1978_PUBLISHED_1978_2002_FLOOR


def test_foreign_registered_does_not_fire_post_1989(
    ruleset: CopyrightRuleSet,
) -> None:
    """A foreign-registered post-2003 work must NOT yield FOREIGN_POST_1989."""
    facts = make_facts(
        pub_year=2010,
        pub_country_code="fr",
        was_registered=True,
    )
    result = assess(facts, ruleset)
    assert result.status is not CopyrightStatus.IN_COPYRIGHT_FOREIGN_POST_1989


def test_foreign_registered_pre_1923_does_not_fire_pd_home_country(
    ruleset: CopyrightRuleSet,
) -> None:
    """A foreign-registered pre-1923 work must NOT yield FOREIGN_IN_HOME_COUNTRY_PD_1996.

    Pinned ``as_of_year=2017`` so the moving-wall short-circuit does not fire.
    The 1931-1963 registered branch requires ``published_between
    [1931, 1963]`` so a 1922 work cannot fire it either; this test
    simply asserts that registration disables the home-country PD path.
    """
    facts = make_facts(
        pub_year=1922,
        pub_country_code="fr",
        was_registered=True,
        as_of_year=2017,
    )
    result = assess(facts, ruleset)
    assert result.status is not CopyrightStatus.PD_FOREIGN_IN_HOME_COUNTRY_PD_1996
