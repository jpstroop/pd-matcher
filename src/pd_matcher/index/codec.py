"""Typed binary codecs for records stored in the LMDB index.

Every value we put into an LMDB sub-DB is serialised with a schema-compiled
``msgspec.msgpack`` codec. Compiling the encoder/decoder against a concrete
:class:`msgspec.Struct` is materially faster than raw msgpack: msgspec walks
the struct's slot layout once at construction time and emits a specialised
serialiser, then reuses it for every record. The codecs themselves are
stateless, so we expose them as module-level singletons and call free
functions that hide them from callers.

Keys also need stable, comparable encodings so LMDB's lexicographic cursor
ordering matches the semantic ordering callers expect:

* Year keys are 2-byte big-endian unsigned integers, which gives a 1923
  bucket that sorts before 1924 with no string padding.
* Renewal-join keys are ``regnum|isoformat(odat)`` UTF-8 strings so they
  collide cleanly with the same key constructed at lookup time from a
  registration record.
"""

from datetime import date

from msgspec.msgpack import Decoder
from msgspec.msgpack import Encoder

from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import NyplRenRecord
from pd_matcher.normalize.registration_numbers import normalize_regnum

_REG_ENCODER: Encoder = Encoder()
_REG_DECODER: Decoder[IndexedNyplRegRecord] = Decoder(IndexedNyplRegRecord)

_REN_ENCODER: Encoder = Encoder()
_REN_DECODER: Decoder[NyplRenRecord] = Decoder(NyplRenRecord)

_UUID_LIST_ENCODER: Encoder = Encoder()
_UUID_LIST_DECODER: Decoder[tuple[str, ...]] = Decoder(tuple[str, ...])


def encode_reg(record: IndexedNyplRegRecord) -> bytes:
    """Serialise an :class:`IndexedNyplRegRecord` to msgpack bytes."""
    return _REG_ENCODER.encode(record)


def decode_reg(blob: bytes) -> IndexedNyplRegRecord:
    """Deserialise msgpack bytes back into an :class:`IndexedNyplRegRecord`."""
    return _REG_DECODER.decode(blob)


def encode_ren(record: NyplRenRecord) -> bytes:
    """Serialise a :class:`NyplRenRecord` to msgpack bytes."""
    return _REN_ENCODER.encode(record)


def decode_ren(blob: bytes) -> NyplRenRecord:
    """Deserialise msgpack bytes back into a :class:`NyplRenRecord`."""
    return _REN_DECODER.decode(blob)


def encode_uuid_list(uuids: tuple[str, ...]) -> bytes:
    """Serialise an ordered tuple of uuid strings to msgpack bytes."""
    return _UUID_LIST_ENCODER.encode(uuids)


def decode_uuid_list(blob: bytes) -> tuple[str, ...]:
    """Deserialise msgpack bytes back into a tuple of uuid strings."""
    return _UUID_LIST_DECODER.decode(blob)


def encode_year_key(year: int) -> bytes:
    """Encode ``year`` as a 2-byte big-endian unsigned integer.

    Big-endian keeps numeric ordering aligned with LMDB's lexicographic
    cursor ordering, so a range scan over the ``reg_by_year`` sub-DB walks
    years in ascending order without further work.
    """
    return year.to_bytes(2, "big")


def decode_year_key(key: bytes) -> int:
    """Decode a 2-byte big-endian year key back to an ``int``."""
    return int.from_bytes(key, "big")


def make_renewal_key(regnum: str, regdate: date | None) -> bytes:
    """Build the join key shared by ``ren_by_oreg`` writers and readers.

    The registration number is canonicalised with :func:`normalize_regnum`
    before assembly so transcription variance (interior spaces, hyphens,
    verbose foreign/interim class phrases) cannot split an otherwise-valid
    join. The same normalizer runs on the renewal ``oreg`` writer and the
    registration ``regnum`` reader, so both sides land on the identical key.
    The date suffix is left untouched: registration id numbers are not unique
    across the catalog's series, so the date remains part of the join key.

    Both sides assemble ``f"{normalize_regnum(regnum)}|{isoformat(regdate) if
    regdate else ''}"``. The ``|`` separator is reserved punctuation that does
    not survive normalization, so it cannot collide with the regnum payload.

    Args:
        regnum: Copyright Office registration number.
        regdate: Original registration date, or ``None`` when absent.

    Returns:
        UTF-8 encoded composite key suitable for use as an LMDB key.
    """
    suffix = regdate.isoformat() if regdate is not None else ""
    return f"{normalize_regnum(regnum)}|{suffix}".encode()


__all__ = [
    "decode_reg",
    "decode_ren",
    "decode_uuid_list",
    "decode_year_key",
    "encode_reg",
    "encode_ren",
    "encode_uuid_list",
    "encode_year_key",
    "make_renewal_key",
]
