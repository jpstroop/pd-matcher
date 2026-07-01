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
* Renewal-join keys are ``regnum|year`` UTF-8 strings so they collide
  cleanly with the same key constructed at lookup time from a registration
  record. The year (not the full date) is the join granularity because
  registration numbers are reused across years but a renewal's ``odat`` and
  its registration's ``reg_date`` routinely disagree by days.
"""

from msgspec.msgpack import Decoder
from msgspec.msgpack import Encoder

from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import NyplRenRecord
from pd_matcher.normalize.registration_numbers import is_multi_regnum
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


def make_renewal_key(regnum: str, year: int | None) -> bytes:
    """Build the join key shared by ``ren_by_oreg`` writers and readers.

    The registration number is canonicalised with :func:`normalize_regnum`
    before assembly so transcription variance (interior spaces, hyphens,
    verbose foreign/interim class phrases) cannot split an otherwise-valid
    join. The same normalizer runs on the renewal ``oreg`` writer and the
    registration ``regnum`` reader, so both sides land on the identical key.

    The suffix is the four-digit **registration year**, not the full ISO date.
    Two-thirds of normalized regnums are reused across years, so the year is
    required to disambiguate the join; but exact-date agreement is not — a
    renewal's ``odat`` and its registration's ``reg_date`` routinely differ by
    days while naming the same registration, and tens of thousands of
    registrations carry a derived year with no ``<regDate>`` at all. Keying on
    the year joins both. The registration side supplies ``reg_year`` (its
    ``regDate → copyDate → pubDate`` fallback) and the renewal side supplies
    ``odat.year``; the two align because ``reg_year`` equals ``reg_date.year``
    whenever a ``<regDate>`` exists.

    Both sides assemble ``f"{normalize_regnum(regnum)}|{year or ''}"``. The
    ``|`` separator is reserved punctuation that does not survive
    normalization, so it cannot collide with the regnum payload.

    Args:
        regnum: Copyright Office registration number.
        year: Registration year — ``odat.year`` on the renewal side,
            ``reg_year`` on the registration side — or ``None`` when the record
            carries neither a date nor a derivable year (it then keys to an
            empty suffix and can never join a renewal).

    Returns:
        UTF-8 encoded composite key suitable for use as an LMDB key.
    """
    suffix = str(year) if year is not None else ""
    return f"{normalize_regnum(regnum)}|{suffix}".encode()


def make_renewal_keys(regnum: str, year: int | None) -> tuple[bytes, ...]:
    """Build every join key a registration or renewal contributes to the join.

    A registered multi-volume whole records several numbers in one ``regnum``
    value (``"A692774 A692775"``); :func:`make_renewal_key` would collapse it
    into a single mashed token (``A692774A692775|…``) that a renewal citing an
    interior number (``A692775``) can never collide with. When
    :func:`is_multi_regnum` recognises such a range, this fans it out into one
    normalized key per listed number so both the ``ren_by_oreg`` writer and the
    registration reader land on the same per-number key. Otherwise it returns a
    one-tuple byte-identical to :func:`make_renewal_key`. The year suffix is
    carried on every key exactly as the single-key path does.

    Args:
        regnum: Copyright Office registration number, possibly a
            space-separated multi-number range.
        year: Registration year, or ``None`` when the record carries neither a
            date nor a derivable year.

    Returns:
        A tuple of UTF-8 encoded composite keys, one per listed number for a
        multi-number range and a single key otherwise.
    """
    if not is_multi_regnum(regnum):
        return (make_renewal_key(regnum, year),)
    suffix = str(year) if year is not None else ""
    return tuple(f"{normalize_regnum(token)}|{suffix}".encode() for token in regnum.split())


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
    "make_renewal_keys",
]
