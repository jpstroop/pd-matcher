"""Tests for :mod:`pd_matcher.match.signals.multipart`."""

from pd_matcher.match.signals.multipart import is_series_level
from pd_matcher.models import MarcRecord


def _marc(
    *,
    extent: str | None = None,
    publication_date_raw: str | None = None,
) -> MarcRecord:
    return MarcRecord(
        control_id="m",
        title="Some title",
        title_main="Some title",
        extent=extent,
        publication_date_raw=publication_date_raw,
    )


def test_bare_v_extent_fires() -> None:
    """AACR2 bare ``"v"`` extent triggers the series-level signal."""
    assert is_series_level(_marc(extent="v")) is True


def test_volumes_extent_fires() -> None:
    """RDA bare ``"volumes"`` extent triggers the series-level signal."""
    assert is_series_level(_marc(extent="volumes")) is True


def test_bare_v_extent_match_is_case_insensitive() -> None:
    """The bare-volume match is case-insensitive."""
    assert is_series_level(_marc(extent="V")) is True
    assert is_series_level(_marc(extent="Volumes")) is True


def test_extent_with_surrounding_whitespace_fires() -> None:
    """Surrounding whitespace on the bare-volume extent is stripped before match."""
    assert is_series_level(_marc(extent="  v  ")) is True


def test_closed_volume_count_does_not_fire() -> None:
    """``"3 v."`` is a closed multi-volume count, not the series-level sentinel."""
    assert is_series_level(_marc(extent="3 v.")) is False


def test_specific_volume_does_not_fire() -> None:
    """``"v. 1"`` is a specific volume marker, not the bare-v series sentinel."""
    assert is_series_level(_marc(extent="v. 1")) is False


def test_open_publication_date_fires() -> None:
    """The ``"[1945-]"`` open-date convention triggers the signal."""
    assert is_series_level(_marc(publication_date_raw="[1945-]")) is True


def test_open_publication_date_with_trailing_space_fires() -> None:
    """``"[1945- ]"`` (trailing space inside brackets) still trips the rule."""
    assert is_series_level(_marc(publication_date_raw="[1945- ]")) is True


def test_truncated_open_publication_date_fires() -> None:
    """``"[1945-"`` (truncated, no closing bracket) is still treated as open."""
    assert is_series_level(_marc(publication_date_raw="[1945-")) is True


def test_closed_bracketed_date_does_not_fire() -> None:
    """``"[1945-1950]"`` is a closed range, not the open-date sentinel."""
    assert is_series_level(_marc(publication_date_raw="[1945-1950]")) is False


def test_plain_year_does_not_fire() -> None:
    """A plain year ``"1945"`` is not an open-date string."""
    assert is_series_level(_marc(publication_date_raw="1945")) is False


def test_bracketed_plain_year_does_not_fire() -> None:
    """``"[1945]"`` (closed bracketed year, no trailing hyphen) does not fire."""
    assert is_series_level(_marc(publication_date_raw="[1945]")) is False


def test_neither_cue_does_not_fire() -> None:
    """A normal closed-monograph MARC record yields ``False``."""
    assert is_series_level(_marc(extent="312 p.", publication_date_raw="1945")) is False


def test_empty_extent_and_empty_date_do_not_fire() -> None:
    """Empty strings on both signal fields yield ``False``."""
    assert is_series_level(_marc(extent="", publication_date_raw="")) is False


def test_none_inputs_do_not_fire() -> None:
    """``None`` on both signal fields yields ``False`` (no spurious matches)."""
    assert is_series_level(_marc(extent=None, publication_date_raw=None)) is False


def test_either_cue_alone_is_sufficient() -> None:
    """The predicate is the disjunction of the extent and date cues."""
    assert is_series_level(_marc(extent="v", publication_date_raw="1945")) is True
    assert is_series_level(_marc(extent="312 p.", publication_date_raw="[1945-]")) is True
