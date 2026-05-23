"""Tests for the uvicorn launch glue (``webui`` marker).

These exercise :func:`pd_groundtruth.review.server.serve` without ever binding
a real socket: the ``uvicorn.run`` call is monkeypatched so the test asserts
only the shutdown-handling contract. ``serve`` is excluded from coverage (it
otherwise just calls uvicorn), so these live under the deselected ``webui``
marker alongside the route tests.
"""

from pathlib import Path

from pytest import MonkeyPatch
from pytest import mark
from pytest import raises

from pd_groundtruth.review import server
from pd_groundtruth.review_db import ReviewDb

pytestmark = mark.webui


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "review.db"
    with ReviewDb.connect(db_path):
        pass
    return db_path


def test_serve_swallows_keyboard_interrupt(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    db_path = _make_db(tmp_path)

    def _raise_interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "uvicorn_run", _raise_interrupt)
    server.serve(db_path, tmp_path / "vault.jsonl", host="127.0.0.1", port=8000)


def test_serve_propagates_non_interrupt_errors(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    db_path = _make_db(tmp_path)

    def _raise_bind_error(*args: object, **kwargs: object) -> None:
        raise OSError("address already in use")

    monkeypatch.setattr(server, "uvicorn_run", _raise_bind_error)
    with raises(OSError, match="address already in use"):
        server.serve(db_path, tmp_path / "vault.jsonl", host="127.0.0.1", port=8000)


def test_serve_rejects_missing_database(tmp_path: Path) -> None:
    with raises(FileNotFoundError, match="review database not found"):
        server.serve(tmp_path / "absent.db", tmp_path / "vault.jsonl", host="127.0.0.1", port=8000)
