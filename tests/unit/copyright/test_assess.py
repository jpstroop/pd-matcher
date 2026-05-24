"""End-to-end tests for :func:`pd_matcher.copyright.assess_record`."""

from hypothesis import given
from hypothesis import strategies as st

from pd_matcher.config.schemas import CopyrightRule
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import PredicateCall
from pd_matcher.copyright import assess_record
from pd_matcher.copyright import default_ruleset
from pd_matcher.copyright.facts import Facts
from pd_matcher.copyright.rules import assess
from pd_matcher.copyright.status import CopyrightStatus
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.result import CandidateMatch
from pd_matcher.match.result import MatchResult
from pd_matcher.models import MarcRecord
from tests.unit.copyright.conftest import AS_OF_YEAR
from tests.unit.copyright.conftest import WIDE_COVERAGE
from tests.unit.copyright.conftest import make_facts


def _registered_match() -> MatchResult:
    """Return a ``MatchResult`` whose ``best`` makes ``was_registered`` true."""
    return MatchResult(
        marc_control_id="marc-1",
        best=CandidateMatch(
            nypl_uuid="nypl-1",
            nypl_year=1950,
            combined=CombinedScore(raw=90.0, calibrated=0.9),
            evidence=(
                Evidence(
                    scorer="title",
                    score=90.0,
                    max=100.0,
                    skipped=False,
                    decisive=False,
                    features=(),
                ),
            ),
            losing_evidence=(),
        ),
        alternates=(),
        candidates_considered=1,
    )


def _marc(
    *,
    publication_year: int | None = 1985,
    country_code: str | None = "nyu",
) -> MarcRecord:
    """Build a minimal MarcRecord for end-to-end assess_record tests."""
    return MarcRecord(
        control_id="marc-1",
        title="T",
        title_main="T",
        publication_year=publication_year,
        country_code=country_code,
    )


def test_assess_record_uses_shipped_ruleset_by_default() -> None:
    """No explicit ruleset -> the shipped Cornell matrix is used.

    A wide :class:`Coverage` is passed so the 1985 no-registration rule
    can fire under the shipped Cornell matrix without being short-
    circuited by the coverage guard.
    """
    marc = _marc(publication_year=1985, country_code="nyu")
    result = assess_record(marc, None, as_of_year=AS_OF_YEAR, coverage=WIDE_COVERAGE)
    assert result.status is CopyrightStatus.PD_US_PUB_NO_REGISTRATION_1978_1989


def test_assess_record_short_circuits_via_moving_wall() -> None:
    """A pre-moving-wall MARC record returns ``PD_BY_AGE_PRE_95_YEARS``."""
    marc = _marc(publication_year=1900)
    result = assess_record(marc, None, as_of_year=AS_OF_YEAR)
    assert result.status is CopyrightStatus.PD_BY_AGE_PRE_95_YEARS
    assert result.matched_rule_name == "moving_wall_short_circuit"


def test_assess_record_defaults_as_of_year_to_current_year() -> None:
    """Omitting ``as_of_year`` falls back to the current year without erroring."""
    marc = _marc(publication_year=1990)
    result = assess_record(marc, None)
    assert isinstance(result.status, CopyrightStatus)


def test_assess_record_accepts_custom_ruleset() -> None:
    """An explicit ``ruleset`` overrides the shipped default."""
    custom = CopyrightRuleSet(
        version="custom",
        rules=[
            CopyrightRule(
                name="catch_all",
                when=[],
                then="UNKNOWN_INSUFFICIENT_DATA",
                explanation="custom",
            ),
        ],
    )
    marc = _marc(publication_year=1985)
    result = assess_record(marc, None, as_of_year=AS_OF_YEAR, ruleset=custom)
    assert result.status is CopyrightStatus.UNKNOWN_INSUFFICIENT_DATA


def test_assess_record_enable_assumptions_flag_threads_through() -> None:
    """The ``enable_assumptions`` flag reaches the rule engine via assess_record."""
    custom = CopyrightRuleSet(
        version="strict",
        rules=[
            CopyrightRule(
                name="needs_notice_and_registration",
                when=[
                    PredicateCall(predicate="country_is_us"),
                    PredicateCall(predicate="was_registered"),
                    PredicateCall(predicate="has_us_notice"),
                ],
                then="IN_COPYRIGHT_REGISTERED_AND_RENEWED",
                explanation="needs notice",
            ),
        ],
    )
    marc = _marc(publication_year=1950)
    match = _registered_match()
    permissive = assess_record(marc, match, as_of_year=AS_OF_YEAR, ruleset=custom)
    strict = assess_record(
        marc,
        match,
        as_of_year=AS_OF_YEAR,
        ruleset=custom,
        enable_assumptions=False,
    )
    assert permissive.status is CopyrightStatus.IN_COPYRIGHT_REGISTERED_AND_RENEWED
    assert strict.status is CopyrightStatus.UNKNOWN_NO_RULE_MATCHED


@given(
    pub_year=st.one_of(st.none(), st.integers(min_value=1500, max_value=2100)),
    country=st.sampled_from(["nyu", "fr", "er", "af", None]),
    registered=st.booleans(),
    renewed=st.booleans(),
)
def test_assess_is_deterministic(
    pub_year: int | None,
    country: str | None,
    registered: bool,
    renewed: bool,
) -> None:
    """Two evaluations of the same (Facts, ruleset) tuple return equal assessments."""
    facts: Facts = make_facts(
        pub_year=pub_year,
        pub_country_code=country,
        was_registered=registered,
        was_renewed=renewed and registered,
        as_of_year=2026,
    )
    rs = default_ruleset()
    first = assess(facts, rs)
    second = assess(facts, rs)
    assert first == second
