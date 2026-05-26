"""Unit tests for the cookie-backed session navigation trail.

These cover the pure :func:`advance` transition (every case in the plan's
behavior table 1-6), the back/forward accessor symmetry, and the cookie
round-trip / malformed-cookie handling (cases 7-10). Cookie tests use the
Starlette primitives directly rather than spinning up a ``TestClient`` —
:func:`read_history` only touches ``request.cookies`` and
:func:`write_history` only calls ``response.set_cookie``, so a constructed
Request/Response is the smallest unit that exercises the real code paths.
"""

from msgspec.json import encode as json_encode
from starlette.requests import Request
from starlette.responses import Response

from pd_groundtruth.review.nav_history import NavHistory
from pd_groundtruth.review.nav_history import advance
from pd_groundtruth.review.nav_history import back_id
from pd_groundtruth.review.nav_history import empty
from pd_groundtruth.review.nav_history import forward_id
from pd_groundtruth.review.nav_history import read_history
from pd_groundtruth.review.nav_history import write_history


def _request_with_cookie(value: str | None) -> Request:
    """Build a minimal Starlette Request carrying the ``pdm_nav`` cookie."""
    headers: list[tuple[bytes, bytes]] = []
    if value is not None:
        headers.append((b"cookie", f"pdm_nav={value}".encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
    }
    return Request(scope)


def test_empty_history_is_zero_state() -> None:
    history = empty()
    assert history.trail == ()
    assert history.cursor == 0


def test_advance_from_empty_records_first_visit() -> None:
    after = advance(empty(), 5)
    assert after.trail == (5,)
    assert after.cursor == 0


def test_advance_refresh_returns_unchanged_history() -> None:
    history = NavHistory(trail=(1, 2, 3), cursor=1)
    assert advance(history, 2) is history


def test_advance_back_step_moves_cursor_left() -> None:
    history = NavHistory(trail=(1, 2, 3), cursor=2)
    after = advance(history, 2)
    assert after.trail == (1, 2, 3)
    assert after.cursor == 1


def test_advance_forward_step_moves_cursor_right() -> None:
    history = NavHistory(trail=(1, 2, 3), cursor=0)
    after = advance(history, 2)
    assert after.trail == (1, 2, 3)
    assert after.cursor == 1


def test_advance_jump_truncates_forward_branch_and_appends() -> None:
    history = NavHistory(trail=(1, 2, 3, 4), cursor=1)
    after = advance(history, 99)
    assert after.trail == (1, 2, 99)
    assert after.cursor == 2


def test_advance_jump_from_tail_appends() -> None:
    history = NavHistory(trail=(1, 2, 3), cursor=2)
    after = advance(history, 4)
    assert after.trail == (1, 2, 3, 4)
    assert after.cursor == 3


def test_advance_at_max_trail_drops_oldest_entries() -> None:
    trail = tuple(range(200))
    history = NavHistory(trail=trail, cursor=199)
    after = advance(history, 999)
    assert len(after.trail) == 200
    assert after.trail[0] == 1
    assert after.trail[-1] == 999
    assert after.cursor == 199


def test_advance_jump_from_middle_at_max_trail_still_caps() -> None:
    trail = tuple(range(200))
    history = NavHistory(trail=trail, cursor=199)
    once = advance(history, 1000)
    twice = advance(once, 1001)
    assert len(twice.trail) == 200
    assert twice.trail[-1] == 1001
    assert twice.cursor == 199


def test_back_id_returns_previous_pair_when_cursor_not_at_head() -> None:
    history = NavHistory(trail=(7, 8, 9), cursor=2)
    assert back_id(history) == 8


def test_back_id_returns_none_when_cursor_at_head() -> None:
    assert back_id(empty()) is None
    assert back_id(NavHistory(trail=(7,), cursor=0)) is None


def test_forward_id_returns_next_pair_when_cursor_not_at_tail() -> None:
    history = NavHistory(trail=(7, 8, 9), cursor=0)
    assert forward_id(history) == 8


def test_forward_id_returns_none_when_cursor_at_tail() -> None:
    assert forward_id(empty()) is None
    assert forward_id(NavHistory(trail=(7,), cursor=0)) is None


def test_write_then_read_round_trips_history() -> None:
    history = NavHistory(trail=(1, 5, 9), cursor=1)
    response = Response()
    write_history(response, history)
    raw = response.headers["set-cookie"]
    cookie_value = raw.split("pdm_nav=", 1)[1].split(";", 1)[0]
    request = _request_with_cookie(cookie_value)
    assert read_history(request) == history


def test_read_history_returns_empty_when_cookie_absent() -> None:
    assert read_history(_request_with_cookie(None)) == empty()


def test_read_history_returns_empty_when_cookie_malformed_json() -> None:
    assert read_history(_request_with_cookie("not-json")) == empty()


def test_read_history_returns_empty_when_cookie_violates_schema() -> None:
    bogus = json_encode({"unexpected": [1, 2, 3]}).decode("utf-8")
    assert read_history(_request_with_cookie(bogus)) == empty()


def test_write_history_sets_session_cookie_with_security_flags() -> None:
    response = Response()
    write_history(response, NavHistory(trail=(1,), cursor=0))
    raw = response.headers["set-cookie"].lower()
    assert "httponly" in raw
    assert "samesite=strict" in raw
    assert "max-age" not in raw
    assert "expires" not in raw
