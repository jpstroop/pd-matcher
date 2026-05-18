"""Tests for :mod:`pd_matcher.workers.events`."""

from pd_matcher.copyright.status import CopyrightStatus
from pd_matcher.workers.events import ProducerHeartbeat
from pd_matcher.workers.events import RecordProcessed
from pd_matcher.workers.events import ShutdownEvent
from pd_matcher.workers.events import WriterHeartbeat
from pd_matcher.workers.events import decode_stats_event
from pd_matcher.workers.events import encode_stats_event


def test_record_processed_roundtrip() -> None:
    event = RecordProcessed(
        status=CopyrightStatus.PD_REGISTERED_NOT_RENEWED,
        confidence=0.84,
        candidates_considered=5,
    )
    decoded = decode_stats_event(encode_stats_event(event))
    assert decoded == event


def test_producer_heartbeat_roundtrip() -> None:
    event = ProducerHeartbeat(records_enqueued=128)
    decoded = decode_stats_event(encode_stats_event(event))
    assert decoded == event


def test_writer_heartbeat_roundtrip() -> None:
    event = WriterHeartbeat(records_written=64)
    decoded = decode_stats_event(encode_stats_event(event))
    assert decoded == event


def test_shutdown_event_roundtrip() -> None:
    event = ShutdownEvent(reason="completed")
    decoded = decode_stats_event(encode_stats_event(event))
    assert decoded == event


def test_events_are_disambiguated_by_tag() -> None:
    """All four event types decode through the same tagged union without collision."""
    blobs = [
        encode_stats_event(
            RecordProcessed(
                status=CopyrightStatus.PD_BY_AGE_PRE_95_YEARS,
                confidence=0.99,
                candidates_considered=1,
            )
        ),
        encode_stats_event(ProducerHeartbeat(records_enqueued=1)),
        encode_stats_event(WriterHeartbeat(records_written=1)),
        encode_stats_event(ShutdownEvent(reason="sigint")),
    ]
    decoded = [decode_stats_event(blob) for blob in blobs]
    assert isinstance(decoded[0], RecordProcessed)
    assert isinstance(decoded[1], ProducerHeartbeat)
    assert isinstance(decoded[2], WriterHeartbeat)
    assert isinstance(decoded[3], ShutdownEvent)
