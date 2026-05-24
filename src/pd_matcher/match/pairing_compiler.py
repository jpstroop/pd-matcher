"""Compile a :class:`PairingConfig` into typed, validated field accessors.

This module is the bridge between configuration and the matching pipeline.
The design boundary is deliberate:

* **Code surfaces raw subfields.** Two finite *raw-field registries*
  (:data:`MARC_FIELDS`, :data:`CCE_FIELDS`) expose each parsed subfield by
  name through an explicit, fully-typed accessor returning
  ``tuple[str, ...]``. Scalar fields wrap to a 0- or 1-tuple; list fields
  pass through. There is no ``getattr`` (which would leak ``Any``); every
  accessor is written out.
* **Configuration composes and pairs them.** A :class:`FieldSpec` names
  one or more registry entries and a *combine op* from a closed
  vocabulary (``first``, ``concat``/``join``). A :class:`PairingSpec`
  routes a composed MARC field against a composed CCE field under a
  scorer group.

:func:`compile_pairings` resolves every name once, at load time, and
raises :class:`~pd_matcher.config.loader.ConfigError` on any unknown name
so typos fail at startup rather than silently producing empty matches.
The result, :class:`CompiledPairings`, holds the per-group accessor pairs
as plain typed callables ready for the hot loop.
"""

from collections.abc import Callable

from msgspec import Struct

from pd_matcher.config.loader import ConfigError
from pd_matcher.config.schemas import FieldSpec
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.config.schemas import PairingSpec
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord

MarcRawAccessor = Callable[[MarcRecord], tuple[str, ...]]
CceRawAccessor = Callable[[IndexedNyplRegRecord], tuple[str, ...]]
MarcAccessor = Callable[[MarcRecord], str | None]
CceAccessor = Callable[[IndexedNyplRegRecord], str | None]
CombineOp = Callable[[tuple[str, ...], str], str | None]


def _scalar(value: str | None) -> tuple[str, ...]:
    """Wrap an optional scalar into a 0- or 1-element tuple."""
    return () if value is None else (value,)


MARC_FIELDS: dict[str, MarcRawAccessor] = {
    "title": lambda marc: _scalar(marc.title),
    "title_main": lambda marc: _scalar(marc.title_main),
    "main_author": lambda marc: _scalar(marc.main_author),
    "statement_of_responsibility": lambda marc: _scalar(marc.statement_of_responsibility),
    "publisher": lambda marc: _scalar(marc.publisher),
    "series_titles": lambda marc: marc.series_titles,
}

CCE_FIELDS: dict[str, CceRawAccessor] = {
    "title": lambda nypl: _scalar(nypl.title),
    "author_name": lambda nypl: _scalar(nypl.author_name),
    "publisher_names": lambda nypl: nypl.publisher_names,
    "claimants": lambda nypl: nypl.claimants,
}


def _combine_first(values: tuple[str, ...], separator: str) -> str | None:
    """Return the first non-empty value, or ``None`` when all are empty."""
    for value in values:
        if value:
            return value
    return None


def _combine_join(values: tuple[str, ...], separator: str) -> str | None:
    """Join non-empty values by ``separator``; ``None`` when all are empty."""
    kept = [value for value in values if value]
    if not kept:
        return None
    return separator.join(kept)


_COMBINE_OPS: dict[str, CombineOp] = {
    "first": _combine_first,
    "concat": _combine_join,
    "join": _combine_join,
}


class CompiledPairing(Struct, frozen=True, forbid_unknown_fields=True):
    """One pairing as a pair of compiled, typed accessors plus its group.

    ``marc_name`` and ``cce_name`` retain the YAML pairing entries' ``marc:``
    and ``cce:`` keys so downstream code (e.g. evidence-source breadcrumbs in
    the review UI) can label which composed-field pair produced a winning
    score, even when the same scorer group has multiple pairings.
    """

    group: str
    marc_name: str
    cce_name: str
    marc_accessor: MarcAccessor
    cce_accessor: CceAccessor


class CompiledPairings(Struct, frozen=True, forbid_unknown_fields=True):
    """All compiled pairings, bucketed by scorer group."""

    title: tuple[CompiledPairing, ...]
    author: tuple[CompiledPairing, ...]
    publisher: tuple[CompiledPairing, ...]


def _compile_marc_field(name: str, spec: FieldSpec) -> MarcAccessor:
    """Compile a MARC :class:`FieldSpec` into a ``str | None`` accessor."""
    accessors = _resolve_marc_accessors(name, spec)
    combine = _COMBINE_OPS[spec.combine]
    separator = spec.separator

    def accessor(marc: MarcRecord) -> str | None:
        values: tuple[str, ...] = ()
        for raw in accessors:
            values = values + raw(marc)
        return combine(values, separator)

    return accessor


def _compile_cce_field(name: str, spec: FieldSpec) -> CceAccessor:
    """Compile a CCE :class:`FieldSpec` into a ``str | None`` accessor."""
    accessors = _resolve_cce_accessors(name, spec)
    combine = _COMBINE_OPS[spec.combine]
    separator = spec.separator

    def accessor(nypl: IndexedNyplRegRecord) -> str | None:
        values: tuple[str, ...] = ()
        for raw in accessors:
            values = values + raw(nypl)
        return combine(values, separator)

    return accessor


def _resolve_marc_accessors(name: str, spec: FieldSpec) -> tuple[MarcRawAccessor, ...]:
    """Resolve a MARC field's raw names; raise on any unknown name."""
    resolved: list[MarcRawAccessor] = []
    for field_name in spec.fields:
        raw = MARC_FIELDS.get(field_name)
        if raw is None:
            raise ConfigError(
                f"marc_fields[{name!r}] references unknown raw MARC field {field_name!r}"
            )
        resolved.append(raw)
    return tuple(resolved)


def _resolve_cce_accessors(name: str, spec: FieldSpec) -> tuple[CceRawAccessor, ...]:
    """Resolve a CCE field's raw names; raise on any unknown name."""
    resolved: list[CceRawAccessor] = []
    for field_name in spec.fields:
        raw = CCE_FIELDS.get(field_name)
        if raw is None:
            raise ConfigError(
                f"cce_fields[{name!r}] references unknown raw CCE field {field_name!r}"
            )
        resolved.append(raw)
    return tuple(resolved)


def _compile_pairing(
    pairing: PairingSpec,
    marc_accessors: dict[str, MarcAccessor],
    cce_accessors: dict[str, CceAccessor],
) -> CompiledPairing:
    """Bind one :class:`PairingSpec` to its compiled accessors."""
    marc_accessor = marc_accessors.get(pairing.marc)
    if marc_accessor is None:
        raise ConfigError(f"pairing references unknown marc field {pairing.marc!r}")
    cce_accessor = cce_accessors.get(pairing.cce)
    if cce_accessor is None:
        raise ConfigError(f"pairing references unknown cce field {pairing.cce!r}")
    return CompiledPairing(
        group=pairing.group,
        marc_name=pairing.marc,
        cce_name=pairing.cce,
        marc_accessor=marc_accessor,
        cce_accessor=cce_accessor,
    )


def compile_pairings(cfg: PairingConfig) -> CompiledPairings:
    """Compile a :class:`PairingConfig` into typed, validated accessors.

    Resolves every ``FieldSpec.fields`` name against the raw-field
    registries and every ``PairingSpec.marc`` / ``.cce`` name against the
    config's field maps, then buckets the compiled pairings by scorer
    group.

    Args:
        cfg: The loaded :class:`PairingConfig`.

    Returns:
        A :class:`CompiledPairings` ready for the matching pipeline.

    Raises:
        ConfigError: If any field or pairing name is unknown.
    """
    marc_accessors = {
        name: _compile_marc_field(name, spec) for name, spec in cfg.marc_fields.items()
    }
    cce_accessors = {name: _compile_cce_field(name, spec) for name, spec in cfg.cce_fields.items()}
    buckets: dict[str, list[CompiledPairing]] = {"title": [], "author": [], "publisher": []}
    for pairing in cfg.pairings:
        compiled = _compile_pairing(pairing, marc_accessors, cce_accessors)
        buckets[pairing.group].append(compiled)
    return CompiledPairings(
        title=tuple(buckets["title"]),
        author=tuple(buckets["author"]),
        publisher=tuple(buckets["publisher"]),
    )


__all__ = [
    "CCE_FIELDS",
    "MARC_FIELDS",
    "CceAccessor",
    "CompiledPairing",
    "CompiledPairings",
    "MarcAccessor",
    "compile_pairings",
]
