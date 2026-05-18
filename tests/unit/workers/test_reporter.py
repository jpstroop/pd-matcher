"""Tests for :class:`pd_matcher.workers.reporter.Reporter`."""

from queue import Queue
from time import sleep

from pd_matcher.copyright.status import CopyrightStatus
from pd_matcher.workers.events import ProducerHeartbeat
from pd_matcher.workers.events import RecordProcessed
from pd_matcher.workers.events import ShutdownEvent
from pd_matcher.workers.events import WriterHeartbeat
from pd_matcher.workers.events import encode_stats_event
from pd_matcher.workers.reporter import Reporter


def test_reporter_aggregates_and_stops_on_shutdown() -> None:
    queue: Queue[bytes] = Queue()
    queue.put(
        encode_stats_event(
            RecordProcessed(
                status=CopyrightStatus.PD_BY_AGE_PRE_95_YEARS,
                confidence=0.9,
                candidates_considered=1,
            )
        )
    )
    queue.put(
        encode_stats_event(
            RecordProcessed(
                status=CopyrightStatus.PD_REGISTERED_NOT_RENEWED,
                confidence=0.8,
                candidates_considered=2,
            )
        )
    )
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
    assert snapshot.by_status == {
        CopyrightStatus.PD_BY_AGE_PRE_95_YEARS.value: 1,
        CopyrightStatus.PD_REGISTERED_NOT_RENEWED.value: 1,
    }


def test_reporter_logs_on_idle_intervals() -> None:
    """The reporter must log progress even while waiting on an empty queue."""
    queue: Queue[bytes] = Queue()
    with Reporter(queue, report_interval_seconds=0.0):
        sleep(0.2)
        queue.put(encode_stats_event(ShutdownEvent(reason="completed")))


def test_reporter_skips_idle_log_when_interval_not_elapsed() -> None:
    """An empty-queue timeout that comes before the interval does NOT log."""
    queue: Queue[bytes] = Queue()
    # A 600-second interval guarantees the idle path runs at least once
    # with the "interval not yet elapsed" branch taken before we feed
    # in the ShutdownEvent.
    with Reporter(queue, report_interval_seconds=600.0):
        sleep(0.2)
        queue.put(encode_stats_event(ShutdownEvent(reason="completed")))
