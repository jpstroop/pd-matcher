"""Tests for :class:`pd_matcher.workers.reporter.RunningTotals`."""

from pd_matcher.workers.events import ProducerHeartbeat
from pd_matcher.workers.events import RecordProcessed
from pd_matcher.workers.events import ShutdownEvent
from pd_matcher.workers.events import WriterHeartbeat
from pd_matcher.workers.reporter import RunningTotals


def test_record_processed_updates_counters() -> None:
    totals = RunningTotals(started_at=0.0)
    totals.apply(RecordProcessed(confidence=0.9, candidates_considered=2))
    totals.apply(RecordProcessed(confidence=0.7, candidates_considered=3))
    assert totals.records_processed == 2


def test_producer_heartbeat_overwrites_enqueued() -> None:
    totals = RunningTotals(started_at=0.0)
    totals.apply(ProducerHeartbeat(records_enqueued=10))
    totals.apply(ProducerHeartbeat(records_enqueued=20))
    assert totals.records_enqueued == 20


def test_writer_heartbeat_overwrites_written() -> None:
    totals = RunningTotals(started_at=0.0)
    totals.apply(WriterHeartbeat(records_written=5))
    totals.apply(WriterHeartbeat(records_written=12))
    assert totals.records_written == 12


def test_shutdown_event_returns_true_and_records_reason() -> None:
    totals = RunningTotals(started_at=0.0)
    stop = totals.apply(ShutdownEvent(reason="sigint"))
    assert stop is True
    assert totals.stop_reason == "sigint"


def test_throughput_returns_zero_when_elapsed_nonpositive() -> None:
    totals = RunningTotals(started_at=10.0)
    totals.records_processed = 100
    assert totals.throughput_per_sec(now=10.0) == 0.0


def test_throughput_per_sec_uses_elapsed_time() -> None:
    totals = RunningTotals(started_at=0.0)
    totals.records_processed = 200
    assert totals.throughput_per_sec(now=10.0) == 20.0


def test_eta_seconds_unknown_when_total_expected_is_none() -> None:
    totals = RunningTotals(started_at=0.0)
    totals.records_processed = 100
    assert totals.eta_seconds(total_expected=None, now=10.0) is None


def test_eta_seconds_unknown_when_already_past_expected() -> None:
    totals = RunningTotals(started_at=0.0)
    totals.records_processed = 200
    assert totals.eta_seconds(total_expected=100, now=10.0) is None


def test_eta_seconds_unknown_when_rate_is_zero() -> None:
    totals = RunningTotals(started_at=0.0)
    totals.records_processed = 0
    assert totals.eta_seconds(total_expected=100, now=10.0) is None


def test_eta_seconds_estimates_remaining_time() -> None:
    totals = RunningTotals(started_at=0.0)
    totals.records_processed = 100
    eta = totals.eta_seconds(total_expected=300, now=10.0)
    assert eta is not None
    assert eta == 20.0


def test_snapshot_exposes_counters() -> None:
    totals = RunningTotals(started_at=0.0)
    totals.records_processed = 1
    snapshot = totals.snapshot()
    assert snapshot.records_processed == 1
    assert snapshot.stop_reason == "running"
