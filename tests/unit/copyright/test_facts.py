"""Tests for :mod:`pd_matcher.copyright.facts`."""

from pytest import raises

from pd_matcher.copyright.facts import Facts
from pd_matcher.copyright.facts import build_facts
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.result import CandidateMatch
from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord


def _marc(
    *,
    publication_year: int | None = 1950,
    country_code: str | None = "nyu",
    publisher: str | None = "Acme Press",
) -> MarcRecord:
    """Build a minimal MarcRecord with optional overrides."""
    return MarcRecord(
        control_id="marc-1",
        title="T",
        title_main="T",
        publication_year=publication_year,
        country_code=country_code,
        language_code="eng",
        publisher=publisher,
    )


def _indexed_nypl(*, was_renewed: bool, claimants: tuple[str, ...] = ()) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="nypl-1",
        title="T",
        was_renewed=was_renewed,
        publisher_names=("National Park Service",),
        claimants=claimants,
    )


def _match_result(confidence: float) -> MatchResult:
    evidence = Evidence(
        scorer="title",
        score=80.0,
        max=100.0,
        skipped=False,
        decisive=False,
        features=(),
    )
    best = CandidateMatch(
        nypl_uuid="nypl-1",
        nypl_year=1950,
        combined=CombinedScore(raw=80.0, calibrated=confidence),
        evidence=(evidence,),
        losing_evidence=(),
    )
    return MatchResult(
        marc_control_id="marc-1",
        best=best,
        alternates=(),
        candidates_considered=1,
    )


def test_facts_is_frozen() -> None:
    """The Facts struct is immutable after construction."""
    facts = Facts(
        pub_year=1950,
        pub_country_code="nyu",
        language_code="eng",
        publisher_text="x",
        was_registered=False,
        was_renewed=False,
        match_confidence=0.0,
        as_of_year=2026,
    )
    with raises(AttributeError):
        setattr(facts, "pub_year", 1999)


def test_facts_has_no_unpublished_or_media_fields() -> None:
    """The struct must not carry Category 1/4/5 fields."""
    facts = Facts(
        pub_year=1950,
        pub_country_code="nyu",
        language_code="eng",
        publisher_text="x",
        was_registered=False,
        was_renewed=False,
        match_confidence=0.0,
        as_of_year=2026,
    )
    for forbidden in (
        "author_death_year",
        "creation_year",
        "is_sound_recording",
        "is_architectural_work",
    ):
        assert not hasattr(facts, forbidden)


def test_build_facts_without_match() -> None:
    """``match=None`` results in unregistered, unrenewed, zero-confidence facts."""
    marc = _marc()
    facts = build_facts(marc, None, as_of_year=2026)
    assert facts.pub_year == 1950
    assert facts.pub_country_code == "nyu"
    assert facts.was_registered is False
    assert facts.was_renewed is False
    assert facts.match_confidence == 0.0
    assert facts.publisher_text == "acme press"


def test_build_facts_with_match_but_no_hydrated_nypl() -> None:
    """A MatchResult without ``matched_nypl`` still flips ``was_registered``."""
    marc = _marc()
    match = _match_result(0.92)
    facts = build_facts(marc, match, as_of_year=2026)
    assert facts.was_registered is True
    assert facts.was_renewed is False
    assert facts.match_confidence == 0.92


def test_build_facts_with_match_and_hydrated_nypl() -> None:
    """Supplying ``matched_nypl`` joins publisher tokens and renewal flag."""
    marc = _marc()
    match = _match_result(0.92)
    nypl = _indexed_nypl(was_renewed=True, claimants=("Estate of John Doe",))
    facts = build_facts(marc, match, as_of_year=2026, matched_nypl=nypl)
    assert facts.was_registered is True
    assert facts.was_renewed is True
    assert facts.publisher_text is not None
    assert "national park service" in facts.publisher_text
    assert "estate of john doe" in facts.publisher_text
    assert "acme press" in facts.publisher_text


def test_build_facts_publisher_text_none_when_no_input() -> None:
    """No MARC publisher and no NYPL hydration -> ``publisher_text=None``."""
    marc = _marc(publisher=None)
    facts = build_facts(marc, None, as_of_year=2026)
    assert facts.publisher_text is None


def test_build_facts_match_with_no_best() -> None:
    """A MatchResult whose ``best`` is None does not flip ``was_registered``."""
    empty_match = MatchResult(
        marc_control_id="marc-1",
        best=None,
        alternates=(),
        candidates_considered=0,
    )
    facts = build_facts(_marc(), empty_match, as_of_year=2026)
    assert facts.was_registered is False
    assert facts.match_confidence == 0.0
