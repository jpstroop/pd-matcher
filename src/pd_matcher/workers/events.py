"""Typed events exchanged across the Phase 6 stats queue.

Workers, the writer process, and the producer publish progress events to
a single stats :class:`multiprocessing.Queue`; the reporter thread (which
runs in the main process) consumes them. Every event is a frozen
:class:`msgspec.Struct` so the wire layout is stable and we get
schema-validated decoding for free. msgpack via :mod:`msgspec.msgpack` is
the cross-process codec — it is faster than pickle, refuses unknown
fields, and produces bytes that ``multiprocessing.Queue`` can ferry
without any pickle reducers.

Two flow events that are emitted by workers / writer / producer (record
processed, heartbeat, shutdown) and one record-keeping event are
defined here. They share a single discriminated union so the reporter
can decode a single byte stream without knowing which event type to
expect ahead of time.
"""

from typing import Final

from msgspec import Struct
from msgspec.msgpack import Decoder
from msgspec.msgpack import Encoder


class RecordProcessed(Struct, frozen=True, forbid_unknown_fields=True, tag="record_processed"):
    """One MARC record was processed by a worker."""

    confidence: float
    candidates_considered: int


class ProducerHeartbeat(Struct, frozen=True, forbid_unknown_fields=True, tag="producer_heartbeat"):
    """Periodic stats from the producer: how many records have been queued."""

    records_enqueued: int


class WriterHeartbeat(Struct, frozen=True, forbid_unknown_fields=True, tag="writer_heartbeat"):
    """Periodic stats from the writer: how many rows have been written."""

    records_written: int


class ShutdownEvent(Struct, frozen=True, forbid_unknown_fields=True, tag="shutdown"):
    """Final event published to the reporter to terminate it."""

    reason: str


StatsEvent = RecordProcessed | ProducerHeartbeat | WriterHeartbeat | ShutdownEvent


_ENCODER: Final[Encoder] = Encoder()
_DECODER: Final[Decoder[StatsEvent]] = Decoder(StatsEvent)


def encode_stats_event(event: StatsEvent) -> bytes:
    """Serialize a :data:`StatsEvent` to msgpack bytes."""
    return _ENCODER.encode(event)


def decode_stats_event(blob: bytes) -> StatsEvent:
    """Deserialize bytes produced by :func:`encode_stats_event`."""
    return _DECODER.decode(blob)


__all__ = [
    "ProducerHeartbeat",
    "RecordProcessed",
    "ShutdownEvent",
    "StatsEvent",
    "WriterHeartbeat",
    "decode_stats_event",
    "encode_stats_event",
]
