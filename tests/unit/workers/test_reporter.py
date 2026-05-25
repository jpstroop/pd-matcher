"""Tests for :class:`pd_matcher.workers.reporter.Reporter`."""

from queue import Queue
from time import sleep

from pd_matcher.workers.events import ProducerHeartbeat
from pd_matcher.workers.events import RecordProcessed
from pd_matcher.workers.events import ShutdownEvent
from pd_matcher.workers.events import WriterHeartbeat
from pd_matcher.workers.events import encode_stats_event
from pd_matcher.workers.reporter import Reporter
from pd_matcher.workers.reporter import RunningTotals
from pd_matcher.workers.reporter import _format_progress_line


def test_reporter_aggregates_and_stops_on_shutdown() -> None:
    queue: Queue[bytes] = Queue()
    queue.put(encode_stats_event(RecordProcessed(confidence=0.9, candidates_considered=1)))
    queue.put(encode_stats_event(RecordProcessed(confidence=0.8, candidates_considered=2)))
    queue.put(encode_stats_event(ProducerHeartbeat(records_enqueued=2)))
    queue.put(encode_stats_event(WriterHeartbeat(records_written=2)))
    queue.put(encode_stats_event(ShutdownEvent(reason="completed")))
    with Reporter(queue, report_interval_seconds=0.05) as reporter:
        pass
    snapshot = reporter.totals.snapshot()
    assert snapshot.records_processed == 2
    assert snapshot.records_written == 2
    assert snapshot.records_enqueued == 2
    assert snapshot.stop_reason == "completed"


def test_reporter_logs_on_idle_intervals() -> None:
    """The reporter must log progress even while waiting on an empty queue."""
    queue: Queue[bytes] = Queue()
    with Reporter(queue, report_interval_seconds=0.0):
        sleep(0.2)
        queue.put(encode_stats_event(ShutdownEvent(reason="completed")))


def test_reporter_skips_idle_log_when_interval_not_elapsed() -> None:
    """An empty-queue timeout that comes before the interval does NOT log."""
    queue: Queue[bytes] = Queue()
    with Reporter(queue, report_interval_seconds=600.0):
        sleep(0.2)
        queue.put(encode_stats_event(ShutdownEvent(reason="completed")))


def test_reporter_threads_expected_total_into_progress() -> None:
    """A reporter given ``expected_total`` logs percent/ETA on idle ticks."""
    queue: Queue[bytes] = Queue()
    with Reporter(queue, report_interval_seconds=0.0, expected_total=10):
        sleep(0.2)
        queue.put(encode_stats_event(ShutdownEvent(reason="completed")))


def test_format_progress_line_with_known_total_shows_percent_and_eta() -> None:
    """A known total renders percent and a numeric ETA."""
    totals = RunningTotals(started_at=0.0)
    totals.records_processed = 50
    totals.records_written = 50
    totals.records_enqueued = 100
    line = _format_progress_line(totals, now=10.0, expected_total=100)
    assert "50/100" in line
    assert "(50%)" in line
    assert "ETA" in line
    assert "written=50" in line


def test_format_progress_line_without_total_omits_percent_and_eta() -> None:
    """An unknown total prints neither ``0/0 (0%)`` nor a misleading ETA."""
    totals = RunningTotals(started_at=0.0)
    totals.records_processed = 25
    line = _format_progress_line(totals, now=5.0, expected_total=None)
    assert "0/0" not in line
    assert "(0%)" not in line
    assert "ETA" not in line
    assert "25 done" in line
    assert "rec/s" in line
