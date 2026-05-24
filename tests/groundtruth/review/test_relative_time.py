"""Unit tests for the ``/labels`` relative-time bucket renderer."""

from datetime import UTC
from datetime import datetime
from datetime import timedelta

from pytest import mark

from pd_groundtruth.review.relative_time import format_relative

_NOW: datetime = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)


def _ago(**delta: float) -> str:
    return (_NOW - timedelta(**delta)).isoformat()


@mark.parametrize(
    ("when", "expected"),
    [
        (_ago(seconds=0), "just now"),
        (_ago(seconds=59), "just now"),
        (_ago(seconds=60), "1m ago"),
        (_ago(minutes=3), "3m ago"),
        (_ago(minutes=59), "59m ago"),
        (_ago(hours=1), "1h ago"),
        (_ago(hours=3), "3h ago"),
        (_ago(hours=23, minutes=59), "23h ago"),
        (_ago(days=1), "1d ago"),
        (_ago(days=5), "5d ago"),
        (_ago(days=13, hours=23), "13d ago"),
        (_ago(days=14), "2w ago"),
        (_ago(days=20, hours=23), "2w ago"),
    ],
)
def test_format_relative_buckets(when: str, expected: str) -> None:
    assert format_relative(when, _NOW) == expected


def test_format_relative_falls_back_to_iso_date_after_three_weeks() -> None:
    past = _NOW - timedelta(days=42)
    assert format_relative(past.isoformat(), _NOW) == past.date().isoformat()


def test_format_relative_future_timestamp_rounds_to_just_now() -> None:
    future = (_NOW + timedelta(seconds=30)).isoformat()
    assert format_relative(future, _NOW) == "just now"


def test_format_relative_handles_naive_timestamp_assuming_now_tz() -> None:
    naive = (_NOW - timedelta(hours=2)).replace(tzinfo=None).isoformat()
    assert format_relative(naive, _NOW) == "2h ago"
