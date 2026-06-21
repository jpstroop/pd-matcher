"""Tests for :mod:`pd_matcher.workers.writer`."""

from collections.abc import Callable
from datetime import date
from functools import partial
from itertools import count
from pathlib import Path

from msgspec.json import decode as json_decode

from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.result import CandidateMatch
from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.output.jsonl_writer import JsonlResultWriter
from pd_matcher.workers.events import WriterHeartbeat
from pd_matcher.workers.events import decode_stats_event
from pd_matcher.workers.messages import WorkerOutput
from pd_matcher.workers.messages import encode_worker_output
from pd_matcher.workers.writer import run_writer_loop
from pd_matcher.workers.writer import writer_main


def _read_records(path: Path) -> list[dict[str, str]]:
    """Decode each non-empty JSONL line into a record dict."""
    return [
        json_decode(line, type=dict[str, str])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _make_payload(control_id: str) -> bytes:
    marc = MarcRecord(control_id=control_id, title="t", title_main="t", publication_year=1940)
    empty_match = MatchResult(
        marc_control_id=control_id,
        best=None,
        alternates=(),
        candidates_considered=0,
    )
    output = WorkerOutput(
        marc=marc,
        match=empty_match,
        matched_nypl=None,
    )
    return encode_worker_output(output)


def _make_matched_payload(control_id: str) -> bytes:
    marc = MarcRecord(control_id=control_id, title="t", title_main="t", publication_year=1940)
    nypl = IndexedNyplRegRecord(
        uuid=f"uuid-{control_id}",
        title="t",
        was_renewed=False,
        reg_date=date(1940, 1, 1),
        reg_year=1940,
    )
    match = MatchResult(
        marc_control_id=control_id,
        best=CandidateMatch(
            nypl_uuid=nypl.uuid,
            nypl_year=1940,
            combined=CombinedScore(raw=90.0, calibrated=0.9),
            evidence=(),
            losing_evidence=(),
        ),
        alternates=(),
        candidates_considered=1,
    )
    output = WorkerOutput(marc=marc, match=match, matched_nypl=nypl)
    return encode_worker_output(output)


def _build_get(blobs: list[bytes | None]) -> Callable[[], bytes | None]:
    iterator = iter(blobs)

    def get() -> bytes | None:
        return next(iterator)

    return get


def test_writer_main_writes_jsonl_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    blobs: list[bytes | None] = [
        _make_payload("m-1"),
        _make_payload("m-2"),
        _make_payload("m-3"),
        None,
    ]
    stats: list[bytes] = []
    written = writer_main(
        output_path=path,
        writer_factory=JsonlResultWriter,
        output_get=_build_get(blobs),
        stats_put=stats.append,
        is_shutdown=lambda: False,
    )
    assert written == 3
    records = _read_records(path)
    assert [record["marc_id"] for record in records] == ["m-1", "m-2", "m-3"]
    final = decode_stats_event(stats[-1])
    assert isinstance(final, WriterHeartbeat)
    assert final.records_written == 3


def test_run_writer_loop_emits_periodic_heartbeats(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    blobs: list[bytes | None] = [
        _make_payload("m-1"),
        _make_payload("m-2"),
        _make_payload("m-3"),
        None,
    ]
    stats: list[bytes] = []
    ticks = count(0, step=10)
    with JsonlResultWriter(path) as writer:
        written = run_writer_loop(
            writer=writer,
            output_get=_build_get(blobs),
            stats_put=stats.append,
            is_shutdown=lambda: False,
            heartbeat_interval_seconds=5.0,
            clock=lambda: float(next(ticks)),
        )
    assert written == 3
    heartbeats: list[WriterHeartbeat] = []
    for blob in stats:
        decoded = decode_stats_event(blob)
        assert isinstance(decoded, WriterHeartbeat)
        heartbeats.append(decoded)
    assert len(heartbeats) >= 2
    assert heartbeats[-1].records_written == 3


def test_run_writer_loop_breaks_when_shutdown_flips(tmp_path: Path) -> None:
    """A flipped shutdown flag terminates the loop after the next write."""
    path = tmp_path / "out.jsonl"
    blobs: list[bytes | None] = [
        _make_payload("m-1"),
        _make_payload("m-2"),
        _make_payload("m-3"),
    ]
    stats: list[bytes] = []
    with JsonlResultWriter(path) as writer:
        written = run_writer_loop(
            writer=writer,
            output_get=_build_get(blobs),
            stats_put=stats.append,
            is_shutdown=lambda: True,
            heartbeat_interval_seconds=5.0,
        )
    records = _read_records(path)
    assert written == 1
    assert len(records) == 1


def test_run_writer_loop_counts_only_real_writes_under_matches_only(tmp_path: Path) -> None:
    """Skipped no-match rows are excluded from the written count and the file."""
    path = tmp_path / "out.jsonl"
    blobs: list[bytes | None] = [
        _make_payload("m-1"),
        _make_matched_payload("m-2"),
        _make_payload("m-3"),
        None,
    ]
    stats: list[bytes] = []
    factory = partial(JsonlResultWriter, matches_only=True)
    written = writer_main(
        output_path=path,
        writer_factory=factory,
        output_get=_build_get(blobs),
        stats_put=stats.append,
        is_shutdown=lambda: False,
    )
    assert written == 1
    records = _read_records(path)
    assert [record["marc_id"] for record in records] == ["m-2"]
    final = decode_stats_event(stats[-1])
    assert isinstance(final, WriterHeartbeat)
    assert final.records_written == 1


def test_run_writer_loop_breaks_on_poison_pill_before_shutdown(tmp_path: Path) -> None:
    """An explicit ``None`` sentinel terminates the loop even when no shutdown is set."""
    path = tmp_path / "out.jsonl"
    blobs: list[bytes | None] = [None]
    stats: list[bytes] = []
    with JsonlResultWriter(path) as writer:
        written = run_writer_loop(
            writer=writer,
            output_get=_build_get(blobs),
            stats_put=stats.append,
            is_shutdown=lambda: False,
        )
    assert written == 0
