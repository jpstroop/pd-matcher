"""Tests for :mod:`pd_matcher.match.pairings`."""

from pd_matcher.match.pairings import publisher_pairings
from pd_matcher.match.pairings import title_pairings
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord


def _marc(
    *,
    title: str = "Primary title",
    publisher: str | None = "Acme",
    series_titles: tuple[str, ...] = (),
) -> MarcRecord:
    return MarcRecord(
        control_id="m",
        title=title,
        publisher=publisher,
        series_titles=series_titles,
    )


def _nypl(
    *,
    title: str = "Other title",
    publisher_names: tuple[str, ...] = ("NYPL Press",),
    claimants: tuple[str, ...] = ("Estate",),
) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="u",
        title=title,
        was_renewed=False,
        publisher_names=publisher_names,
        claimants=claimants,
    )


def test_title_pairings_returns_primary_only_when_no_series() -> None:
    """Without series titles we get exactly one pairing."""
    pairings = title_pairings(_marc(), _nypl(title="NYPL"))
    assert pairings == (("Primary title", "NYPL"),)


def test_title_pairings_includes_first_two_series_titles() -> None:
    """At most three pairings total are returned."""
    marc = _marc(series_titles=("Series A", "Series B", "Series C"))
    pairings = title_pairings(marc, _nypl(title="NYPL"))
    assert pairings == (
        ("Primary title", "NYPL"),
        ("Series A", "NYPL"),
        ("Series B", "NYPL"),
    )


def test_publisher_pairings_returns_publisher_names_and_claimants() -> None:
    """We get two pairings: one against publisher_names, one against claimants."""
    pairings = publisher_pairings(
        _marc(publisher="Acme"),
        _nypl(publisher_names=("Acme Press",), claimants=("Acme Inc.",)),
    )
    assert pairings == (("Acme", "Acme Press"), ("Acme", "Acme Inc."))


def test_publisher_pairings_passes_none_marc_publisher_through() -> None:
    """A None MARC publisher is preserved so the scorer can skip cleanly."""
    pairings = publisher_pairings(_marc(publisher=None), _nypl())
    assert pairings[0][0] is None
    assert pairings[1][0] is None
