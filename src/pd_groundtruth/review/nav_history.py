"""Cookie-backed session navigation trail for the review UI.

The reviewer's intent — "page back and forth through what I just looked at" —
is genuinely a per-browser-session concept, not anything reconstructable from
the database. ``review_pair.id`` ordering is unstable across queue rebuilds
(vault-backfilled pairs land at scattered ids) and walking labeled history
globally leaks pairs from prior sessions the user doesn't remember acting on.
A session cookie carrying the exact trail of pair ids visited this session
sidesteps both problems and naturally extends to multi-user (each browser
already has its own cookie).

The state is a small frozen :class:`NavHistory` — a tuple of pair ids and a
cursor — and the transitions in :func:`advance` collapse four cases (refresh,
back step, forward step, jump) onto that one type. The cookie is HTTP-only,
SameSite=strict, and has no ``max_age`` / ``expires``, making it a session
cookie that browsers clear on close.
"""

from msgspec import DecodeError
from msgspec import Struct
from msgspec.json import decode as json_decode
from msgspec.json import encode as json_encode
from starlette.requests import Request
from starlette.responses import Response

_COOKIE_NAME: str = "pdm_nav"
_MAX_TRAIL: int = 200


class NavHistory(Struct, frozen=True, forbid_unknown_fields=True):
    """One browser session's trail of viewed pair ids and current cursor.

    Invariant: ``trail`` empty implies ``cursor == 0``; otherwise
    ``0 <= cursor < len(trail)``. ``trail[cursor]`` is always the pair the
    user is presently looking at, so back/forward derive from neighboring
    indices.
    """

    trail: tuple[int, ...]
    cursor: int


_EMPTY: NavHistory = NavHistory(trail=(), cursor=0)


def empty() -> NavHistory:
    """Return the zero-state history used when no cookie is present."""
    return _EMPTY


def advance(history: NavHistory, pair_id: int) -> NavHistory:
    """Apply one navigation event and return the resulting history.

    Cases (precedence as listed):

    * **Refresh** — the visited pair equals ``trail[cursor]``: unchanged.
    * **Back step** — the visited pair equals ``trail[cursor - 1]``: cursor
      moves one left, preserving the forward suffix.
    * **Forward step** — the visited pair equals ``trail[cursor + 1]``:
      cursor moves one right, preserving the trail.
    * **Jump** — anything else (including the empty trail and the first
      visit): truncate the trail at ``cursor + 1`` so the forward branch is
      discarded, append the new pair, and advance the cursor onto it. If the
      result exceeds ``_MAX_TRAIL``, drop the oldest entries from the front
      and shift the cursor accordingly so it still points at the new tail.
    """
    trail = history.trail
    cursor = history.cursor
    if trail and trail[cursor] == pair_id:
        return history
    if cursor > 0 and trail[cursor - 1] == pair_id:
        return NavHistory(trail=trail, cursor=cursor - 1)
    if cursor + 1 < len(trail) and trail[cursor + 1] == pair_id:
        return NavHistory(trail=trail, cursor=cursor + 1)
    new_trail = (*trail[: cursor + 1], pair_id) if trail else (pair_id,)
    new_cursor = len(new_trail) - 1
    overflow = len(new_trail) - _MAX_TRAIL
    if overflow > 0:
        new_trail = new_trail[overflow:]
        new_cursor -= overflow
    return NavHistory(trail=new_trail, cursor=new_cursor)


def back_id(history: NavHistory) -> int | None:
    """Return the pair id one step before the cursor, or ``None`` at the head."""
    if history.cursor > 0:
        return history.trail[history.cursor - 1]
    return None


def forward_id(history: NavHistory) -> int | None:
    """Return the pair id one step after the cursor, or ``None`` at the tail."""
    if history.cursor + 1 < len(history.trail):
        return history.trail[history.cursor + 1]
    return None


def read_history(request: Request) -> NavHistory:
    """Decode the ``pdm_nav`` cookie into a :class:`NavHistory`.

    A missing cookie, malformed JSON, or a payload that violates the struct's
    schema all collapse silently to :func:`empty`. The cookie is reviewer
    convenience, not authoritative data — a single corrupt value must not 500
    the page.
    """
    raw = request.cookies.get(_COOKIE_NAME)
    if raw is None:
        return _EMPTY
    try:
        return json_decode(raw.encode("utf-8"), type=NavHistory)
    except DecodeError, ValueError:
        return _EMPTY


def write_history(response: Response, history: NavHistory) -> None:
    """Persist ``history`` to the response's ``pdm_nav`` cookie.

    Set without ``max_age`` / ``expires`` so the browser treats it as a
    session cookie that clears on close. ``httponly`` blocks JS access (the
    nav is purely server-side) and ``samesite=strict`` keeps the trail from
    leaking across origins.
    """
    payload = json_encode(history).decode("utf-8")
    response.set_cookie(
        _COOKIE_NAME,
        payload,
        httponly=True,
        samesite="strict",
    )


__all__ = [
    "NavHistory",
    "advance",
    "back_id",
    "empty",
    "forward_id",
    "read_history",
    "write_history",
]
