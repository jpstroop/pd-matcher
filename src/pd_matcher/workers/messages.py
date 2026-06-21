"""Per-record output messages flowing from workers to the writer process.

Workers encode one :class:`WorkerOutput` per processed MARC record and
push it on the output queue; the writer process decodes the same struct
and hands it to a :class:`pd_matcher.output.jsonl_writer.ResultWriter`.

Keeping the on-wire payload as a single msgspec Struct lets us add audit
fields (matched NYPL UUID, year buckets considered, etc.) without
revisiting the queue boundary. Encoding via :mod:`msgspec.msgpack` is
faster than pickle and refuses unknown fields, so a worker built against
an older schema cannot silently feed malformed data into a newer writer.
"""

from typing import Final

from msgspec import Struct
from msgspec.msgpack import Decoder
from msgspec.msgpack import Encoder

from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord


class WorkerOutput(Struct, frozen=True, forbid_unknown_fields=True):
    """One processed-record payload ferried from a worker to the writer.

    ``match`` is always populated by the worker — the matcher pipeline
    returns a :class:`MatchResult` with ``best=None`` for records that
    failed to clear the floor, rather than ``None`` itself. The writer
    treats ``match.best is None`` as the empty-match signal.
    """

    marc: MarcRecord
    match: MatchResult
    matched_nypl: IndexedNyplRegRecord | None


_ENCODER: Final[Encoder] = Encoder()
_DECODER: Final[Decoder[WorkerOutput]] = Decoder(WorkerOutput)


def encode_worker_output(output: WorkerOutput) -> bytes:
    """Serialize a :class:`WorkerOutput` to msgpack bytes."""
    return _ENCODER.encode(output)


def decode_worker_output(blob: bytes) -> WorkerOutput:
    """Deserialize bytes produced by :func:`encode_worker_output`."""
    return _DECODER.decode(blob)


__all__ = [
    "WorkerOutput",
    "decode_worker_output",
    "encode_worker_output",
]
