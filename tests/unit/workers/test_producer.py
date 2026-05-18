"""Tests for :mod:`pd_matcher.workers.producer`."""

from collections.abc import Iterator
from itertools import count

from pytest import raises

from pd_matcher.models import MarcRecord
from pd_matcher.workers.events import ProducerHeartbeat
from pd_matcher.workers.events import decode_stats_event
from pd_matcher.workers.producer import decode_batch
from pd_matcher.workers.producer import encode_batch
from pd_matcher.workers.producer import iter_decoded_batches
from pd_matcher.workers.producer import run_producer


def _make_records(n: int) -> list[MarcRecord]:
    return [MarcRecord(control_id=f"m-{i}", title=f"t{i}") for i in range(n)]


def test_encode_and_decode_batch_roundtrip() -> None:
    records = tuple(_make_records(3))
    assert decode_batch(encode_batch(records)) == records


def test_iter_decoded_batches_returns_tuples() -> None:
    a = tuple(_make_records(2))
    b = tuple(_make_records(3))
    blobs = [encode_batch(a), encode_batch(b)]
    decoded = list(iter_decoded_batches(blobs))
    assert decoded == [a, b]


def test_run_producer_rejects_zero_batch_size() -> None:
    with raises(ValueError, match="batch_size must be >= 1"):
        run_producer(
            iter(_make_records(1)),
            input_put=lambda _: None,
            stats_put=lambda _: None,
            is_shutdown=lambda: False,
            batch_size=0,
        )


def test_run_producer_batches_records_in_fixed_size_groups() -> None:
    inputs: list[bytes] = []
    stats: list[bytes] = []
    records = _make_records(5)
    enqueued = run_producer(
        iter(records),
        input_put=inputs.append,
        stats_put=stats.append,
        is_shutdown=lambda: False,
        batch_size=2,
    )
    decoded = [decode_batch(blob) for blob in inputs]
    assert enqueued == 5
    assert len(decoded[0]) == 2
    assert len(decoded[1]) == 2
    assert len(decoded[2]) == 1
    # Final heartbeat is always emitted.
    final = decode_stats_event(stats[-1])
    assert isinstance(final, ProducerHeartbeat)
    assert final.records_enqueued == 5


def test_run_producer_emits_periodic_heartbeats() -> None:
    inputs: list[bytes] = []
    stats: list[bytes] = []
    records = _make_records(10)
    ticks = count(0, step=10)
    run_producer(
        iter(records),
        input_put=inputs.append,
        stats_put=stats.append,
        is_shutdown=lambda: False,
        batch_size=2,
        heartbeat_interval_seconds=5.0,
        clock=lambda: float(next(ticks)),
    )
    heartbeats: list[ProducerHeartbeat] = []
    for blob in stats:
        decoded = decode_stats_event(blob)
        assert isinstance(decoded, ProducerHeartbeat)
        heartbeats.append(decoded)
    enqueued_seq = [hb.records_enqueued for hb in heartbeats]
    assert enqueued_seq[-1] == 10
    assert len(enqueued_seq) >= 2


def test_run_producer_stops_on_shutdown_between_records() -> None:
    inputs: list[bytes] = []
    stats: list[bytes] = []
    records = _make_records(10)
    triggered = {"value": False}

    def is_shutdown() -> bool:
        # Signal shutdown after the first input goes through.
        if len(inputs) >= 1:
            triggered["value"] = True
        return triggered["value"]

    enqueued = run_producer(
        iter(records),
        input_put=inputs.append,
        stats_put=stats.append,
        is_shutdown=is_shutdown,
        batch_size=2,
    )
    assert enqueued < 10


def test_run_producer_drops_buffered_records_on_shutdown_at_end() -> None:
    """A trailing partial batch is NOT flushed when shutdown has fired."""
    inputs: list[bytes] = []
    stats: list[bytes] = []
    records = _make_records(3)
    triggered = {"value": False}

    def is_shutdown() -> bool:
        return triggered["value"]

    def src() -> Iterator[MarcRecord]:
        yield from records
        triggered["value"] = True

    enqueued = run_producer(
        src(),
        input_put=inputs.append,
        stats_put=stats.append,
        is_shutdown=is_shutdown,
        batch_size=2,
    )
    # Only one full batch (2) makes it; the trailing 1 is dropped.
    assert enqueued == 2
