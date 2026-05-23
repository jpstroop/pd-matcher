"""Render an ISO-8601 timestamp as a compact relative-time string.

Used by the ``/labels`` table view to label each row's ``labeled_at`` column.
Buckets coarsen with age so the recent activity that matters for spot-checking
stays human-scaled (``"3m ago"``, ``"5h ago"``) while older rows fall back to
the absolute date once relative phrasing stops carrying information.

``now`` is passed in by the caller so tests can pin the clock without reaching
for ``freezegun`` or monkey-patching ``datetime``.
"""

from datetime import datetime
from datetime import timedelta

_MINUTE: int = 60
_HOUR: int = 60 * _MINUTE
_DAY: int = 24 * _HOUR
_TWO_WEEKS: timedelta = timedelta(days=14)
_THREE_WEEKS: timedelta = timedelta(days=21)


def format_relative(iso_timestamp: str, now: datetime) -> str:
    """Return a short relative-time string for ``iso_timestamp`` vs. ``now``.

    Buckets, in order:
      * under 60 s     → ``"just now"``
      * under 60 min   → ``"{n}m ago"``
      * under 24 h     → ``"{n}h ago"``
      * under 14 days  → ``"{n}d ago"``
      * under 21 days  → ``"{n}w ago"`` (caps at ``"3w ago"``)
      * otherwise      → the ISO date ``YYYY-MM-DD``

    Future timestamps (clock skew, replay) round to ``"just now"``. ``now``
    must be timezone-aware; the parsed timestamp is converted to ``now``'s
    timezone before subtraction so the delta is on a consistent basis.
    """
    parsed = datetime.fromisoformat(iso_timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=now.tzinfo)
    delta = now - parsed
    if delta.total_seconds() < _MINUTE:
        return "just now"
    seconds = int(delta.total_seconds())
    if seconds < _HOUR:
        return f"{seconds // _MINUTE}m ago"
    if seconds < _DAY:
        return f"{seconds // _HOUR}h ago"
    if delta < _TWO_WEEKS:
        return f"{delta.days}d ago"
    if delta < _THREE_WEEKS:
        return f"{delta.days // 7}w ago"
    return parsed.date().isoformat()


__all__ = [
    "format_relative",
]
