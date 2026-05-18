"""MARC-record producer that batches records and feeds the input queue.

The producer streams :class:`pd_matcher.models.MarcRecord` instances from
:func:`iter_marc_records`, groups them into fixed-size tuples, and
publishes each batch on the input :class:`multiprocessing.Queue`. It
runs in the main process — :func:`iter_marc_records` is the bottleneck
and does not share state with anyone, so there is no parallelism win
from a separate producer process, and staying in main keeps the
orchestrator straightforward.

A :class:`ProducerHeartbeat` is emitted on the stats queue every
``heartbeat_interval_seconds`` so the reporter can show queueing
progress without inspecting the input queue's internal state.
"""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from time import monotonic
from typing import Final

from msgspec import Struct
from msgspec.msgpack import Decoder
from msgspec.msgpack import Encoder
from structlog import get_logger

from pd_matcher.models import MarcRecord
from pd_matcher.workers.events import ProducerHeartbeat
from pd_matcher.workers.events import encode_stats_event

_LOGGER = get_logger(__name__)


class _MarcBatch(Struct, frozen=True, forbid_unknown_fields=True):
    """Internal wrapper used to serialize a tuple of :class:`MarcRecord`."""

    records: tuple[MarcRecord, ...]


_BATCH_ENCODER: Final[Encoder] = Encoder()
_BATCH_DECODER: Final[Decoder[_MarcBatch]] = Decoder(_MarcBatch)


def encode_batch(batch: tuple[MarcRecord, ...]) -> bytes:
    """Serialize a batch of records to msgpack bytes for the input queue."""
    return _BATCH_ENCODER.encode(_MarcBatch(records=batch))


def decode_batch(blob: bytes) -> tuple[MarcRecord, ...]:
    """Deserialize a batch previously produced by :func:`encode_batch`."""
    return _BATCH_DECODER.decode(blob).records


def run_producer(
    records: Iterable[MarcRecord],
    *,
    input_put: Callable[[bytes], None],
    stats_put: Callable[[bytes], None],
    is_shutdown: Callable[[], bool],
    batch_size: int,
    heartbeat_interval_seconds: float = 5.0,
    clock: Callable[[], float] = monotonic,
) -> int:
    """Stream ``records`` into ``input_put`` in batches of ``batch_size``.

    Args:
        records: Iterable yielding :class:`MarcRecord` instances. Typically
            the :func:`pd_matcher.parsers.marc.iter_marc_records` generator.
        input_put: Callable that pushes one msgpack-encoded batch onto the
            input queue. Usually ``input_queue.put``.
        stats_put: Callable that pushes a stats event blob onto the stats
            queue. Usually ``stats_queue.put``.
        is_shutdown: Zero-arg callable returning ``True`` when the
            shutdown event has been signaled. Checked between batches.
        batch_size: Records per emitted batch. Must be ``>= 1``.
        heartbeat_interval_seconds: Minimum wall-clock seconds between
            :class:`ProducerHeartbeat` emissions.
        clock: Monotonic clock for unit tests that need deterministic time.

    Returns:
        The total number of records that were enqueued. Includes records
        in the final (possibly smaller-than-``batch_size``) flush; excludes
        records that were drained from the source but never queued because
        ``is_shutdown`` flipped between dequeue and enqueue.

    Raises:
        ValueError: If ``batch_size < 1``.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1 (got {batch_size!r})")
    enqueued = 0
    last_heartbeat = clock()
    buffer: list[MarcRecord] = []
    for record in records:
        if is_shutdown():
            break
        buffer.append(record)
        if len(buffer) >= batch_size:
            input_put(encode_batch(tuple(buffer)))
            enqueued += len(buffer)
            buffer.clear()
            now = clock()
            if now - last_heartbeat >= heartbeat_interval_seconds:
                stats_put(encode_stats_event(ProducerHeartbeat(records_enqueued=enqueued)))
                last_heartbeat = now
    if buffer and not is_shutdown():
        input_put(encode_batch(tuple(buffer)))
        enqueued += len(buffer)
    stats_put(encode_stats_event(ProducerHeartbeat(records_enqueued=enqueued)))
    _LOGGER.info("producer.complete", records_enqueued=enqueued)
    return enqueued


def iter_decoded_batches(
    encoded_batches: Iterable[bytes],
) -> Iterator[tuple[MarcRecord, ...]]:
    """Decode an iterable of batch blobs back into tuples of records.

    Helper used by tests and by the worker loop's batch-handling logic;
    isolating the decode here keeps :class:`Decoder` instances internal.
    """
    for blob in encoded_batches:
        yield decode_batch(blob)


__all__ = [
    "decode_batch",
    "encode_batch",
    "iter_decoded_batches",
    "run_producer",
]
