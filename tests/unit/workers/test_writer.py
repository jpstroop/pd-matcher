"""Tests for :mod:`pd_matcher.workers.writer`."""

from collections.abc import Callable
from csv import DictReader
from itertools import count
from pathlib import Path

from pd_matcher.copyright.assessment import CopyrightAssessment
from pd_matcher.copyright.status import CopyrightStatus
from pd_matcher.match.result import MatchResult
from pd_matcher.models import MarcRecord
from pd_matcher.output.csv_writer import CsvResultWriter
from pd_matcher.workers.events import WriterHeartbeat
from pd_matcher.workers.events import decode_stats_event
from pd_matcher.workers.messages import WorkerOutput
from pd_matcher.workers.messages import encode_worker_output
from pd_matcher.workers.writer import run_writer_loop
from pd_matcher.workers.writer import writer_main


def _make_payload(control_id: str) -> bytes:
    marc = MarcRecord(control_id=control_id, title="t", publication_year=1940)
    assessment = CopyrightAssessment(
        status=CopyrightStatus.UNKNOWN_INSUFFICIENT_DATA,
        matched_rule_name=None,
        explanation="",
        assumptions=(),
    )
    empty_match = MatchResult(
        marc_control_id=control_id,
        best=None,
        alternates=(),
        candidates_considered=0,
    )
    output = WorkerOutput(
        marc=marc,
        match=empty_match,
        assessment=assessment,
        matched_nypl=None,
    )
    return encode_worker_output(output)


def _build_get(blobs: list[bytes | None]) -> Callable[[], bytes | None]:
    iterator = iter(blobs)

    def get() -> bytes | None:
        return next(iterator)

    return get


def test_writer_main_writes_csv_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    blobs: list[bytes | None] = [
        _make_payload("m-1"),
        _make_payload("m-2"),
        _make_payload("m-3"),
        None,
    ]
    stats: list[bytes] = []
    written = writer_main(
        output_path=path,
        writer_factory=CsvResultWriter,
        output_get=_build_get(blobs),
        stats_put=stats.append,
        is_shutdown=lambda: False,
    )
    assert written == 3
    rows = list(DictReader(path.open(encoding="utf-8")))
    assert [row["marc_id"] for row in rows] == ["m-1", "m-2", "m-3"]
    # The final heartbeat is always emitted, regardless of cadence.
    final = decode_stats_event(stats[-1])
    assert isinstance(final, WriterHeartbeat)
    assert final.records_written == 3


def test_run_writer_loop_emits_periodic_heartbeats(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    blobs: list[bytes | None] = [
        _make_payload("m-1"),
        _make_payload("m-2"),
        _make_payload("m-3"),
        None,
    ]
    stats: list[bytes] = []
    ticks = count(0, step=10)
    with CsvResultWriter(path) as writer:
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
    # Periodic emissions plus a final flush.
    assert len(heartbeats) >= 2
    assert heartbeats[-1].records_written == 3


def test_run_writer_loop_breaks_when_shutdown_flips(tmp_path: Path) -> None:
    """A flipped shutdown flag terminates the loop after the next write."""
    path = tmp_path / "out.csv"
    blobs: list[bytes | None] = [
        _make_payload("m-1"),
        _make_payload("m-2"),
        _make_payload("m-3"),
    ]
    stats: list[bytes] = []
    with CsvResultWriter(path) as writer:
        written = run_writer_loop(
            writer=writer,
            output_get=_build_get(blobs),
            stats_put=stats.append,
            is_shutdown=lambda: True,
            heartbeat_interval_seconds=5.0,
        )
    with path.open(encoding="utf-8") as fp:
        rows = list(DictReader(fp))
    assert written == 1
    assert len(rows) == 1


def test_run_writer_loop_breaks_on_poison_pill_before_shutdown(tmp_path: Path) -> None:
    """An explicit ``None`` sentinel terminates the loop even when no shutdown is set."""
    path = tmp_path / "out.csv"
    blobs: list[bytes | None] = [None]
    stats: list[bytes] = []
    with CsvResultWriter(path) as writer:
        written = run_writer_loop(
            writer=writer,
            output_get=_build_get(blobs),
            stats_put=stats.append,
            is_shutdown=lambda: False,
        )
    assert written == 0
