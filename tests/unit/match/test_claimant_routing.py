"""Tests for :mod:`pd_matcher.match.claimant_routing`."""

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.claimant_routing import RoutingDecision
from pd_matcher.match.claimant_routing import compute_routing
from pd_matcher.match.claimant_routing import is_blank_publisher
from pd_matcher.match.claimant_routing import value_key
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.name import score_author
from pd_matcher.models import MarcRecord
from pd_matcher.normalize.stemming import stemmer_for
from pd_matcher.normalize.stopwords import load_stopwords


def _config(floor: float) -> MatchingConfig:
    return MatchingConfig(
        title_weight=0.50,
        author_weight=0.20,
        publisher_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        extent_weight=0.0,
        volume_weight=0.0,
        year_window=0,
        min_combined_score=70.0,
        claimant_routing_floor=floor,
    )


def _context(floor: float) -> ScorerContext:
    """Return a context whose lone author token ``vale`` scores 0.6 normalized."""
    author_idf = IdfTable(
        document_count=10,
        default_idf=4.0,
        source_hash="test",
        language="eng",
        idf={"vale": 2.4},
    )
    generic_idf = IdfTable(
        document_count=10,
        default_idf=4.0,
        source_hash="test",
        language="eng",
        idf={"knopf": 3.9, "albuquerque": 3.9, "jane": 1.2},
    )
    return ScorerContext(
        language="eng",
        stopwords=load_stopwords("eng"),
        stemmer=stemmer_for("eng"),
        idf=generic_idf,
        author_idf=author_idf,
        publisher_idf=generic_idf,
        config=_config(floor),
    )


def test_value_key_is_order_and_case_insensitive() -> None:
    """The routing key is a normalized token set, ignoring order and case."""
    assert value_key("Jane Albuquerque") == value_key("albuquerque, jane")
    assert value_key("") == frozenset()


def test_is_blank_publisher() -> None:
    """A CCE record is blank-publisher only when every publisher name is empty."""
    assert is_blank_publisher(()) is True
    assert is_blank_publisher(("",)) is True
    assert is_blank_publisher(("Random House",)) is False


def test_no_shared_value_does_not_fire() -> None:
    """No routing when publisher and claimant carry different values."""
    marc = MarcRecord(control_id="m", title="t", title_main="t")
    decision = compute_routing(marc, ("Random House",), ("Jane Albuquerque",), _context(0.7))
    assert decision == RoutingDecision(author_routed=frozenset(), publisher_routed=frozenset())
    assert decision.fired is False


def test_routes_shared_value_to_author() -> None:
    """A person echoed into the publisher slot routes to the author group."""
    marc = MarcRecord(
        control_id="m",
        title="t",
        title_main="t",
        main_author="Albuquerque, Jane",
        publisher="Random House",
    )
    decision = compute_routing(marc, ("Jane Albuquerque",), ("Jane Albuquerque",), _context(0.7))
    key = value_key("Jane Albuquerque")
    assert decision.author_routed == frozenset({key})
    assert decision.publisher_routed == frozenset()
    assert decision.fired is True


def test_routes_shared_value_to_publisher() -> None:
    """A corporate self-publisher echoed into the claimant slot routes to publisher."""
    marc = MarcRecord(
        control_id="m",
        title="t",
        title_main="t",
        main_author="Smith, John",
        publisher="Knopf",
    )
    decision = compute_routing(marc, ("Knopf",), ("Knopf",), _context(0.7))
    key = value_key("Knopf")
    assert decision.publisher_routed == frozenset({key})
    assert decision.author_routed == frozenset()


def test_tie_routes_to_author() -> None:
    """When both fields score equally the value routes to the author group."""
    marc = MarcRecord(
        control_id="m",
        title="t",
        title_main="t",
        main_author="Albuquerque",
        publisher="Albuquerque",
    )
    decision = compute_routing(marc, ("Albuquerque",), ("Albuquerque",), _context(0.7))
    assert decision.author_routed == frozenset({value_key("Albuquerque")})
    assert decision.publisher_routed == frozenset()


def test_sub_floor_winner_does_not_route() -> None:
    """A shared value matching neither MARC field is left unrouted (mismatch survives)."""
    marc = MarcRecord(
        control_id="m",
        title="t",
        title_main="t",
        main_author="Smith, John",
        publisher="Random House",
    )
    decision = compute_routing(marc, ("Jane Albuquerque",), ("Jane Albuquerque",), _context(0.7))
    assert decision.fired is False


def test_empty_and_untokenizable_values_are_ignored() -> None:
    """Empty strings and punctuation-only values drop out before key comparison."""
    marc = MarcRecord(
        control_id="m",
        title="t",
        title_main="t",
        main_author="Smith, John",
        publisher="Knopf",
    )
    decision = compute_routing(marc, ("", ",.-", "Knopf"), (",.-", "Knopf"), _context(0.7))
    assert decision.publisher_routed == frozenset({value_key("Knopf")})
    assert decision.author_routed == frozenset()


def test_floor_gates_the_route_between_0_5_and_0_7() -> None:
    """The shared value routes at floor 0.5 but not at floor 0.7 (0.5 <= score < 0.7)."""
    marc = MarcRecord(
        control_id="m",
        title="t",
        title_main="t",
        main_author="vale extra",
        publisher="zzz",
    )
    score = score_author("vale extra", "vale", _context(0.5)).normalized
    assert 0.5 <= score < 0.7
    routed_low = compute_routing(marc, ("vale",), ("vale",), _context(0.5))
    routed_high = compute_routing(marc, ("vale",), ("vale",), _context(0.7))
    assert routed_low.author_routed == frozenset({value_key("vale")})
    assert routed_high.fired is False
