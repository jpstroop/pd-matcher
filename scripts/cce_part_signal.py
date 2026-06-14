"""Size a CCE-side "part-of-a-larger-work" detector for issue #82 (Phase 1 prep).

The maintainer's redirected approach: the cleanest whole/part signal lives on
the CCE side — "does this registration look like part of a larger work" —
detectable by regex over the CCE record's OWN ``title`` + ``notes`` + ``desc``.
A CCE registration is self-referential (it describes the work being
registered), so a designator in its text is citation-clean, unlike a MARC 500
note which routinely leaks volume citations to OTHER works.

Throwaway investigate-first measurement script. NOT shipped; ``scripts/`` is
gitignored from the published package (a ``!`` exception lets the maintainer
commit this proof). It does NOT modify anything under ``src/`` and writes
nothing under ``data/``; the vault is read-only.

TWO ANALYSES.

**Part 1 — MARC note noise by source tag.** The parsed ``MarcRecord.notes``
flattens MARC tags {500, 502, 505, 520} into one tuple (the parser drops the
tag). To attribute the volume-designator noise seen in Phase 0's
``signal_in_unread_field`` table to its SOURCE tag, this re-reads the RAW
MARCXML for each MARC in the whole/part slice and, for every note carrying a
``_PART_NUMBER_RE`` designator, records which 5xx tag it came from. Within 500,
it splits citation-style notes (lead phrases like "Reprinted from", "Abstracted
in") from contents-style notes (multiple ``v.N`` / ``pt.N`` items = this work's
own volumes). This tests whether 500 is salvageable with a citation-lead-phrase
filter or hopeless.

**Part 2 — the CCE part detector.** Builds a candidate detector
``_cce_looks_like_part(cce)`` over CCE ``title`` / each ``note`` / ``desc``,
starting from volume.py's ``_detect_part`` / ``_PART_NUMBER_RE`` and adding
series-statement context patterns seen in the real data (German half-volume
ordinals ``1. hälfte`` / ``1. Halbbd.``, plural ``Vols. 3-4``, trailing
series-comma designators ``, v.3`` / ``, Bd.40``, parenthetical series
``(The reference shelf, vol. VII, no. 6)``, volume ranges ``v.20-21``). It
measures coverage over the whole/part cases (by category + overall) and the
false-positive rate over a seeded clean-negative sample (verdict ``match``,
untagged, keyword-free notes — standalone single-work matches whose CCE reg
should NOT look like a part).

Usage:
    pdm run python scripts/cce_part_signal.py \\
        > docs/findings/cce_part_signal_2026-06-13.md

    # Smoke run capping the slice (report is flagged):
    pdm run python scripts/cce_part_signal.py --limit 40 > /tmp/cce_smoke.md
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from random import Random
from re import IGNORECASE
from re import Pattern
from re import compile as re_compile
from sys import stderr
from typing import Final

from lxml.etree import _Element
from lxml.etree import iterparse

from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.vault_pair_resolver import iter_pool_shards
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.scorers.volume import _PART_NUMBER_RE
from pd_matcher.match.scorers.volume import _detect_part
from pd_matcher.models import IndexedNyplRegRecord

_VAULT_PATH: Final[Path] = Path("data/label_vault.jsonl")
_POOL_PATH: Final[Path] = Path("data/candidates")
_INDEX_PATH: Final[Path] = Path("caches/cce.lmdb")
_PROGRESS_LOG: Final[Path] = Path("/tmp/agent-progress.log")

_CAT_MARC_WHOLE: Final[str] = "marc_whole_cce_part"
_CAT_CCE_WHOLE: Final[str] = "cce_whole_marc_part"
_WHOLE_PART_CATEGORIES: Final[frozenset[str]] = frozenset(
    {_CAT_MARC_WHOLE, _CAT_CCE_WHOLE}
)

_VERDICT_MATCH: Final[str] = "match"

# Clean-negative sampling: deterministic seed + sample size.
_FP_SAMPLE_SEED: Final[int] = 8224
_FP_SAMPLE_SIZE: Final[int] = 200

_MAX_REPRESENTATIVE_ROWS: Final[int] = 25
_MAX_FP_EXAMPLES: Final[int] = 30
_RAW_TEXT_TRUNCATE: Final[int] = 90
_UUID_TRUNCATE: Final[int] = 8

# Source-field labels for the CCE detector's per-field attribution.
_SRC_TITLE: Final[str] = "cce.title"
_SRC_NOTES: Final[str] = "cce.notes"
_SRC_DESC: Final[str] = "cce.desc"

# MARC namespace + tags for the raw-XML re-read (Part 1). The parser flattens
# {500, 502, 505, 520} into MarcRecord.notes, so the tag is only recoverable
# from the raw document.
_MARC_NS: Final[str] = "http://www.loc.gov/MARC21/slim"
_RECORD_TAG: Final[str] = f"{{{_MARC_NS}}}record"
_CONTROLFIELD_TAG: Final[str] = f"{{{_MARC_NS}}}controlfield"
_DATAFIELD_TAG: Final[str] = f"{{{_MARC_NS}}}datafield"
_SUBFIELD_TAG: Final[str] = f"{{{_MARC_NS}}}subfield"
_NOTE_TAGS: Final[tuple[str, ...]] = ("500", "502", "505", "520")
_TAG_500: Final[str] = "500"

# Whole/part keyword surface used to define the clean-negative pool (a note with
# any of these is NOT clean and is excluded from the FP sample). Mirrors the
# Phase-0 script's keyword set.
_KEYWORD_RE: Final[Pattern[str]] = re_compile(
    r"\bv\.?\b"
    r"|vol"
    r"|\bpt\.?\b"
    r"|part"
    r"|tome"
    r"|tomo"
    r"|\bbd\.?\b"
    r"|band"
    r"|teil"
    r"|\bt\.?\s*\d"
    r"|heft"
    r"|fasc"
    r"|livre"
    r"|libro"
    r"|complete"
    r"|collected"
    r"|selected"
    r"|\bser\.?\b"
    r"|aufl"
    r"|course\s+\d",
    IGNORECASE,
)

# Citation lead phrases that mark a MARC 500 note as referring to ANOTHER work
# (the noise the maintainer wants to keep out of the parsed notes signal). A 500
# note opening with one of these whose designator is a citation, not this work's
# own volume count.
_CITATION_LEAD_RE: Final[Pattern[str]] = re_compile(
    r"^\s*(?:"
    r"abstracted\s+in"
    r"|reprinted\s+from"
    r"|reprint\s+of"
    r"|first\s+published\s+in"
    r"|issued\s+as"
    r"|issued\s+in"
    r"|indexed\s+in"
    r"|reviewed\s+in"
    r"|translation\s+of"
    r"|translated\s+from"
    r"|originally\s+published"
    r"|extract(?:ed)?\s+from"
    r"|offprint"
    r"|published\s+in"
    r")",
    IGNORECASE,
)

# A contents-style 500/505 note enumerates THIS work's own volumes: two or more
# distinct ``v.N`` / ``pt.N`` items. Counting designator hits separates a
# multi-volume contents listing from a single stray citation number.
_DESIGNATOR_ITEM_RE: Final[Pattern[str]] = re_compile(
    r"\b(?:v(?:ol(?:ume)?)?|pt|part|bk|book|bd|t)\.?\s*\d", IGNORECASE
)

# --- Part 2: additive CCE part patterns (beyond volume.py's _PART_NUMBER_RE) ---
# Each carries a label so coverage can report which pattern fired. Recall-first:
# precision is measured against the clean-negative sample, not assumed.

# German half-volume ordinal: "1. hälfte", "2. Halbbd.", "1. Halbband".
_GERMAN_HALF_RE: Final[Pattern[str]] = re_compile(
    r"\b\d+\.\s*(?:h(?:ä|ae)lfte|halbbd|halbband)\b", IGNORECASE
)
# Plural volume designator: "Vols. 3-4", "vols 3", "Bde. 1-2" — the singular
# _PART_NUMBER_RE misses the trailing 's'.
_PLURAL_VOLS_RE: Final[Pattern[str]] = re_compile(
    r"\b(?:vols|bde|tt|pts)\.?\s*\[?\s*[ivxlcdm\d]", IGNORECASE
)
# Trailing series-comma designator: "<series name>, v.3", ", Bd.40", ", vol. VII",
# ", no. 6", ", Heft 2". A kind prefix is REQUIRED after the comma to avoid
# matching bare "<title>, 86"-style numbers (high false-positive risk).
_SERIES_TRAIL_RE: Final[Pattern[str]] = re_compile(
    r",\s*(?:v(?:ol)?\.?|bd\.?|t\.?|h(?:eft)?\.?|no\.?|nr\.?)\s*"
    r"([ivxlcdm]+|\d+)\b",
    IGNORECASE,
)
# Parenthetical series with a designator inside: "(The reference shelf, vol. VII,
# no. 6)", "(... 12. bd. )".
_PAREN_SERIES_RE: Final[Pattern[str]] = re_compile(
    r"\([^)]*\b(?:v(?:ol)?\.?|bd\.?|t\.?|no\.?|heft)\s*([ivxlcdm]+|\d+)[^)]*\)",
    IGNORECASE,
)
# Volume range: "v.20-21", "vol. 3-4", "Bd. 1-2".
_VOLUME_RANGE_RE: Final[Pattern[str]] = re_compile(
    r"\b(?:v(?:ol)?|bd|t)\.?\s*\d+\s*-\s*\d+\b", IGNORECASE
)

_ADDITIVE_PATTERNS: Final[tuple[tuple[str, Pattern[str]], ...]] = (
    ("german_half", _GERMAN_HALF_RE),
    ("plural_vols", _PLURAL_VOLS_RE),
    ("volume_range", _VOLUME_RANGE_RE),
    ("series_trailing_comma", _SERIES_TRAIL_RE),
    ("parenthetical_series", _PAREN_SERIES_RE),
)

_PATTERN_BASE: Final[str] = "base_part_number_re"


def _progress(message: str) -> None:
    """Emit a one-line milestone to stderr and the shared progress log."""
    print(message, file=stderr, flush=True)
    with _PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def _cce_field_match(value: str | None) -> str | None:
    """Return the firing pattern label for ``value`` (one CCE field), or ``None``.

    The base ``_detect_part`` (volume.py's ``_PART_NUMBER_RE``) is tried first
    so coverage attributable to the EXISTING regex is separated from the
    additive series-context patterns. ``_is_multivolume_whole`` is NOT consulted
    here: a multi-volume whole is the WHOLE, not the part, and folding it in
    would conflate the two cardinalities.
    """
    if not value:
        return None
    if _detect_part(value) is not None:
        return _PATTERN_BASE
    for label, pattern in _ADDITIVE_PATTERNS:
        if pattern.search(value) is not None:
            return label
    return None


@dataclass(frozen=True, slots=True)
class CceDetection:
    """The outcome of running ``_cce_looks_like_part`` on one CCE record."""

    hit: bool
    pattern: str | None
    source_field: str | None
    matched_text: str | None


def _cce_looks_like_part(cce: IndexedNyplRegRecord) -> CceDetection:
    """Return whether ``cce`` looks like part of a larger work, and how.

    Scans the CCE record's OWN ``title``, then each ``note``, then ``desc`` —
    the self-referential fields. The first firing field wins; its pattern label
    and source-field name are reported so precision risk can be attributed to a
    specific pattern and field. Title is scanned first because it is the
    highest-precision CCE part signal in the real data; ``desc`` last because it
    is mostly a page-count string.
    """
    title_hit = _cce_field_match(cce.title)
    if title_hit is not None:
        return CceDetection(
            hit=True,
            pattern=title_hit,
            source_field=_SRC_TITLE,
            matched_text=cce.title,
        )
    for note in cce.notes:
        note_hit = _cce_field_match(note)
        if note_hit is not None:
            return CceDetection(
                hit=True, pattern=note_hit, source_field=_SRC_NOTES, matched_text=note
            )
    desc_hit = _cce_field_match(cce.desc)
    if desc_hit is not None:
        return CceDetection(
            hit=True, pattern=desc_hit, source_field=_SRC_DESC, matched_text=cce.desc
        )
    return CceDetection(hit=False, pattern=None, source_field=None, matched_text=None)


# --------------------------------------------------------------------------- #
# Part 1 — raw MARCXML re-read for per-tag note attribution.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RawNote:
    """One raw MARC 5xx note with its source tag, for designator attribution."""

    tag: str
    text: str


def _subfield_a_texts(field_elem: _Element) -> list[str]:
    """Collect all ``$a`` subfield text nodes within a datafield element."""
    out: list[str] = []
    for sub in field_elem.iterfind(_SUBFIELD_TAG):
        if sub.get("code") == "a" and sub.text is not None:
            out.append(sub.text)
    return out


def _raw_notes_for_record(record_elem: _Element) -> tuple[str, list[RawNote]] | None:
    """Return ``(control_id, raw 5xx notes)`` for one raw ``<record>`` element."""
    control_id: str | None = None
    notes: list[RawNote] = []
    for child in record_elem:
        if child.tag == _CONTROLFIELD_TAG:
            if child.get("tag") == "001" and child.text is not None:
                control_id = child.text.strip() or None
            continue
        if child.tag != _DATAFIELD_TAG:
            continue
        tag = child.get("tag")
        if tag in _NOTE_TAGS and tag is not None:
            for text in _subfield_a_texts(child):
                stripped = text.strip()
                if stripped:
                    notes.append(RawNote(tag=tag, text=stripped))
    if control_id is None:
        return None
    return control_id, notes


def _raw_notes_by_control_id(
    pool: Path, wanted: set[str]
) -> dict[str, list[RawNote]]:
    """Map ``control_id -> raw 5xx notes`` for every wanted id, by tag.

    Streams the same ``<lang>/*.xml`` shards the parser uses with
    ``lxml.iterparse`` and explicit element clearing, but keeps each note's
    SOURCE TAG (which the production parser discards). Stops early once every
    wanted id is resolved.
    """
    if not wanted:
        return {}
    found: dict[str, list[RawNote]] = {}
    remaining = set(wanted)
    for shard in iter_pool_shards(pool):
        context = iterparse(str(shard), events=("end",), tag=_RECORD_TAG)
        for _event, elem in context:
            result = _raw_notes_for_record(elem)
            if result is not None:
                control_id, notes = result
                if control_id in remaining:
                    found[control_id] = notes
                    remaining.discard(control_id)
            elem.clear()
            previous = elem.getprevious()
            while previous is not None:
                del previous.getparent()[0]
                previous = elem.getprevious()
            if not remaining:
                del context
                return found
        del context
    return found


def _is_citation_500(note: str) -> bool:
    """Return ``True`` when a 500 note opens with a citation lead phrase."""
    return _CITATION_LEAD_RE.search(note) is not None


def _is_contents_500(note: str) -> bool:
    """Return ``True`` when a 500/505 note enumerates 2+ of this work's volumes."""
    return len(_DESIGNATOR_ITEM_RE.findall(note)) >= 2


@dataclass(slots=True)
class MarcTagNoiseReport:
    """Part-1 aggregate: designator-bearing MARC notes by source tag."""

    records_with_raw_notes: int = 0
    designator_notes_by_tag: Counter[str] = field(default_factory=Counter)
    notes_by_tag: Counter[str] = field(default_factory=Counter)
    citation_500: int = 0
    contents_500: int = 0
    other_500: int = 0
    citation_examples: list[RawNote] = field(default_factory=list)
    contents_examples: list[RawNote] = field(default_factory=list)
    other_examples: list[RawNote] = field(default_factory=list)


def _build_marc_tag_noise(
    raw_by_id: dict[str, list[RawNote]], slice_ids: list[str]
) -> MarcTagNoiseReport:
    """Tabulate designator-bearing MARC notes by source tag for the slice."""
    report = MarcTagNoiseReport()
    for control_id in slice_ids:
        notes = raw_by_id.get(control_id)
        if not notes:
            continue
        report.records_with_raw_notes += 1
        for note in notes:
            report.notes_by_tag[note.tag] += 1
            if _PART_NUMBER_RE.search(note.text) is None:
                continue
            report.designator_notes_by_tag[note.tag] += 1
            if note.tag != _TAG_500:
                continue
            if _is_citation_500(note.text):
                report.citation_500 += 1
                if len(report.citation_examples) < _MAX_FP_EXAMPLES:
                    report.citation_examples.append(note)
            elif _is_contents_500(note.text):
                report.contents_500 += 1
                if len(report.contents_examples) < _MAX_FP_EXAMPLES:
                    report.contents_examples.append(note)
            else:
                report.other_500 += 1
                if len(report.other_examples) < _MAX_FP_EXAMPLES:
                    report.other_examples.append(note)
    return report


# --------------------------------------------------------------------------- #
# Part 2 — CCE detector coverage + false-positive rate.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CoverageCase:
    """One whole/part case run through the CCE part detector."""

    marc_control_id: str
    nypl_uuid: str
    category: str
    detection: CceDetection


@dataclass(slots=True)
class CoverageReport:
    """Part-2 coverage: detector hit-rate over the whole/part cases."""

    by_category_total: Counter[str] = field(default_factory=Counter)
    by_category_hit: Counter[str] = field(default_factory=Counter)
    pattern_counts: Counter[str] = field(default_factory=Counter)
    field_counts: Counter[str] = field(default_factory=Counter)
    cases: list[CoverageCase] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FpCase:
    """One clean-negative sample run through the CCE part detector."""

    marc_control_id: str
    nypl_uuid: str
    detection: CceDetection


@dataclass(slots=True)
class FpReport:
    """Part-2 false-positive rate over the seeded clean-negative sample."""

    sample_size: int = 0
    pool_size: int = 0
    hits: int = 0
    pattern_counts: Counter[str] = field(default_factory=Counter)
    field_counts: Counter[str] = field(default_factory=Counter)
    examples: list[FpCase] = field(default_factory=list)


@dataclass(slots=True)
class StudyResult:
    """Everything the report renders."""

    tagged_count: int = 0
    resolved_tagged: int = 0
    missing_in_index: int = 0
    coverage: CoverageReport = field(default_factory=CoverageReport)
    fp: FpReport = field(default_factory=FpReport)
    marc_noise: MarcTagNoiseReport = field(default_factory=MarcTagNoiseReport)
    limit: int | None = None


def _note_is_clean(entry: VaultEntry) -> bool:
    """Return ``True`` when the entry's free-text note has no whole/part keyword."""
    if not entry.note:
        return True
    return _KEYWORD_RE.search(entry.note) is None


def _select_clean_negatives(
    entries: dict[tuple[str, str], VaultEntry],
) -> list[VaultEntry]:
    """Return verdict-``match``, untagged, keyword-free-note entries, deterministically.

    These are standalone single-work matches: a correct MARC↔CCE linkage with no
    whole/part complication. Their CCE registration should NOT look like a part,
    so any detector hit here is a false positive. Ordered by the stable
    ``(marc_control_id, nypl_uuid)`` key before sampling for determinism.
    """
    pool: list[VaultEntry] = []
    for entry in entries.values():
        if entry.verdict != _VERDICT_MATCH:
            continue
        if _WHOLE_PART_CATEGORIES.intersection(entry.categories):
            continue
        if not _note_is_clean(entry):
            continue
        pool.append(entry)
    pool.sort(key=lambda e: (e.marc_control_id, e.nypl_uuid))
    return pool


def run_study(
    entries: dict[tuple[str, str], VaultEntry], limit: int | None
) -> StudyResult:
    """Resolve the slices, run both analyses, and aggregate everything."""
    result = StudyResult(limit=limit)

    tagged: list[VaultEntry] = [
        entry
        for entry in entries.values()
        if _WHOLE_PART_CATEGORIES.intersection(entry.categories)
    ]
    tagged.sort(key=lambda e: (e.marc_control_id, e.nypl_uuid))
    if limit is not None:
        tagged = tagged[:limit]
    result.tagged_count = len(tagged)

    clean_pool = _select_clean_negatives(entries)
    result.fp.pool_size = len(clean_pool)
    rng = Random(_FP_SAMPLE_SEED)
    sample_size = min(_FP_SAMPLE_SIZE, len(clean_pool))
    if limit is not None:
        sample_size = min(sample_size, limit)
    fp_sample = rng.sample(clean_pool, sample_size) if sample_size else []
    fp_sample.sort(key=lambda e: (e.marc_control_id, e.nypl_uuid))
    result.fp.sample_size = len(fp_sample)

    _progress(
        f"slices selected: tagged={len(tagged)} "
        f"clean_pool={len(clean_pool)} fp_sample={len(fp_sample)}"
    )

    with NyplIndexLookup(_INDEX_PATH) as lookup:
        _run_coverage(tagged, lookup, result)
        _progress(
            f"coverage done: resolved={result.resolved_tagged} "
            f"hits={sum(result.coverage.by_category_hit.values())}"
        )
        _run_false_positives(fp_sample, lookup, result)
        _progress(
            f"fp done: sample={result.fp.sample_size} hits={result.fp.hits}"
        )

    slice_marc_ids: set[str] = {entry.marc_control_id for entry in tagged}
    raw_by_id = _raw_notes_by_control_id(_POOL_PATH, slice_marc_ids)
    _progress(f"raw MARC notes resolved for {len(raw_by_id)} records")
    ordered_ids = sorted(slice_marc_ids)
    result.marc_noise = _build_marc_tag_noise(raw_by_id, ordered_ids)
    _progress("marc tag-noise tabulated")

    return result


def _run_coverage(
    tagged: list[VaultEntry], lookup: NyplIndexLookup, result: StudyResult
) -> None:
    """Run the CCE detector over the tagged whole/part cases."""
    for entry in tagged:
        cce = lookup.get_registration(entry.nypl_uuid)
        if cce is None:
            result.missing_in_index += 1
            continue
        result.resolved_tagged += 1
        category = _primary_category(entry)
        detection = _cce_looks_like_part(cce)
        result.coverage.by_category_total[category] += 1
        if detection.hit:
            result.coverage.by_category_hit[category] += 1
            if detection.pattern is not None:
                result.coverage.pattern_counts[detection.pattern] += 1
            if detection.source_field is not None:
                result.coverage.field_counts[detection.source_field] += 1
        result.coverage.cases.append(
            CoverageCase(
                marc_control_id=entry.marc_control_id,
                nypl_uuid=entry.nypl_uuid,
                category=category,
                detection=detection,
            )
        )


def _run_false_positives(
    fp_sample: list[VaultEntry], lookup: NyplIndexLookup, result: StudyResult
) -> None:
    """Run the CCE detector over the clean-negative sample."""
    for entry in fp_sample:
        cce = lookup.get_registration(entry.nypl_uuid)
        if cce is None:
            continue
        detection = _cce_looks_like_part(cce)
        if not detection.hit:
            continue
        result.fp.hits += 1
        if detection.pattern is not None:
            result.fp.pattern_counts[detection.pattern] += 1
        if detection.source_field is not None:
            result.fp.field_counts[detection.source_field] += 1
        if len(result.fp.examples) < _MAX_FP_EXAMPLES:
            result.fp.examples.append(
                FpCase(
                    marc_control_id=entry.marc_control_id,
                    nypl_uuid=entry.nypl_uuid,
                    detection=detection,
                )
            )


def _primary_category(entry: VaultEntry) -> str:
    """Return the whole/part category for an entry (marc_whole wins if both)."""
    if _CAT_MARC_WHOLE in entry.categories:
        return _CAT_MARC_WHOLE
    return _CAT_CCE_WHOLE


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #


def _truncate(text: str | None, length: int) -> str:
    """Return ``text`` collapsed and clipped to ``length`` with an ellipsis."""
    if not text:
        return "—"
    collapsed = " ".join(text.split())
    if len(collapsed) <= length:
        return collapsed
    return collapsed[: length - 1] + "…"


def _print_limit_warning(limit: int | None) -> None:
    """Emit a prominent SMOKE banner when a ``--limit`` was active."""
    if limit is None:
        return
    print(
        f"> ⚠️ **SMOKE RUN — `--limit {limit}` was active.** Only the first "
        f"{limit} tagged whole/part cases (and a sample capped at {limit}) were "
        "processed; every count below is from a truncated set and is NOT a real "
        "finding. Re-run WITHOUT `--limit` for the production report.\n"
    )


def _print_method(result: StudyResult) -> None:
    """Emit the Method section."""
    print("## Method\n")
    print(
        "**Question.** Issue #82's redirected approach: the cleanest whole/part "
        "signal is on the CCE side — *does this registration look like part of a "
        "larger work?* — detectable by regex over the CCE record's OWN `title` + "
        "`notes` + `desc`. A CCE record is self-referential (it describes the "
        "registered work), so a designator in its text is citation-clean, unlike "
        "a MARC 500 note that leaks volume citations to OTHER works.\n"
    )
    print(
        "**Part 1 — MARC note noise by source tag.** The parser flattens MARC "
        "tags {500, 502, 505, 520} into one `MarcRecord.notes` tuple and drops "
        "the tag, so this RE-READS the raw MARCXML (`data/candidates/<lang>/"
        "*.xml`) for every MARC in the whole/part slice via `lxml.iterparse`, "
        "keeping each note's source tag. For every note carrying a "
        "`volume._PART_NUMBER_RE` designator it records the tag; within 500 it "
        "splits citation-style notes (lead phrases `Reprinted from`, `Abstracted "
        "in`, `First published in`, `Reprint of`, `Issued as/in`, `Indexed in`, "
        "`Reviewed in`, `Translation of`, …) from contents-style notes (≥2 "
        "`v.N`/`pt.N` items = this work's own volumes).\n"
    )
    print(
        "**Part 2 — CCE part detector.** `_cce_looks_like_part(cce)` scans the "
        "CCE `title`, then each `note`, then `desc` (first firing field wins). "
        "Each field is tested by `volume._detect_part` (the existing "
        "`_PART_NUMBER_RE`) FIRST, then additive series-context patterns: "
        "`german_half` (`1. hälfte` / `1. Halbbd.`), `plural_vols` (`Vols. 3-4`), "
        "`volume_range` (`v.20-21`), `series_trailing_comma` (`, v.3` / `, Bd.40` "
        "/ `, vol. VII` — a kind prefix is REQUIRED after the comma so bare "
        "`<title>, 86` numbers do not match), and `parenthetical_series` (`(The "
        "reference shelf, vol. VII, no. 6)`). `_is_multivolume_whole` is NOT "
        "consulted — a multi-volume whole is the WHOLE, not the part.\n"
    )
    print(
        "**Coverage** is the detector hit-rate over the tagged whole/part cases "
        f"(`{_CAT_MARC_WHOLE}` / `{_CAT_CCE_WHOLE}`), by category and overall.\n"
    )
    print(
        "**False-positive rate.** The clean-negative pool is every vault entry "
        f"with verdict `{_VERDICT_MATCH}` that is NOT tagged a whole/part "
        "category AND whose free-text `note` carries no whole/part keyword — "
        "standalone single-work matches whose CCE reg should NOT look like a "
        f"part. A deterministic seeded sample (`Random({_FP_SAMPLE_SEED})`, "
        f"n={_FP_SAMPLE_SIZE}, key-sorted before and after sampling) is run "
        "through the detector; any hit is a false positive.\n"
    )


def _print_part1(result: StudyResult) -> None:
    """Emit the Part-1 MARC tag-noise tables and verdict."""
    noise = result.marc_noise
    print("## Part 1 — MARC note noise by source tag\n")
    print(
        f"Over the {result.tagged_count} tagged whole/part MARC records, "
        f"{noise.records_with_raw_notes} had raw 5xx notes. Counts below are "
        "designator-bearing notes (a `_PART_NUMBER_RE` hit) by source tag, with "
        "the total note count per tag for context.\n"
    )
    print("| tag | designator-bearing notes | all notes |")
    print("|:---|---:|---:|")
    for tag in _NOTE_TAGS:
        print(
            f"| {tag} | {noise.designator_notes_by_tag[tag]} "
            f"| {noise.notes_by_tag[tag]} |"
        )
    total_desig = sum(noise.designator_notes_by_tag.values())
    print(f"| **all** | **{total_desig}** | "
          f"**{sum(noise.notes_by_tag.values())}** |")
    print()

    print("### Within tag 500: citation-style vs contents-style\n")
    print(
        "A 500 note opening with a citation lead phrase refers to ANOTHER work "
        "(noise). A 500/505-style note enumerating ≥2 `v.N`/`pt.N` items lists "
        "THIS work's own volumes (salvageable signal). `other` is neither — a "
        "single designator with no citation lead.\n"
    )
    print("| 500 class | count |")
    print("|:---|---:|")
    print(f"| citation-style (noise) | {noise.citation_500} |")
    print(f"| contents-style (own volumes) | {noise.contents_500} |")
    print(f"| other (single, no lead) | {noise.other_500} |")
    print()

    _print_note_examples("Citation-style 500 examples (noise)", noise.citation_examples)
    _print_note_examples(
        "Contents-style 500 examples (own volumes)", noise.contents_examples
    )
    _print_note_examples("Other 500 examples (single, no lead)", noise.other_examples)

    _print_part1_verdict(noise)


def _print_note_examples(heading: str, examples: list[RawNote]) -> None:
    """Emit a labelled list of raw-note examples."""
    print(f"**{heading}**\n")
    if not examples:
        print("_None in this run._\n")
        return
    for note in examples:
        print(f"- `{note.tag}` {_truncate(note.text, _RAW_TEXT_TRUNCATE)}")
    print()


def _print_part1_verdict(noise: MarcTagNoiseReport) -> None:
    """Emit the programmatic verdict on 500 salvageability."""
    print("### Verdict — is MARC 500 salvageable?\n")
    total_500 = noise.citation_500 + noise.contents_500 + noise.other_500
    if total_500 == 0:
        print(
            "_No designator-bearing 500 notes in this run; no verdict._\n"
        )
        return
    citation_share = noise.citation_500 / total_500
    designator_total = sum(noise.designator_notes_by_tag.values())
    share_500 = (
        noise.designator_notes_by_tag[_TAG_500] / designator_total
        if designator_total
        else 0.0
    )
    print(
        f"- Tag 500 carries {share_500:.0%} of all designator-bearing notes in "
        "the slice.\n"
        f"- Within 500, {citation_share:.0%} are citation-style (noise about "
        f"another work), {noise.contents_500 / total_500:.0%} are contents-style "
        f"(this work's own volumes), {noise.other_500 / total_500:.0%} other.\n"
    )
    if citation_share >= 0.5:
        print(
            "**500 is the noise source, and the citation lead-phrase filter "
            "isolates most of it.** The majority of designator-bearing 500 notes "
            "are citations to other works; a lead-phrase exclusion would strip "
            "most noise while keeping the contents-style notes that genuinely "
            "describe the registered work. This corroborates the maintainer's "
            "theory that the parsed-notes designator leakage is 500-driven, and "
            "argues for filtering 500 rather than dropping it — but the CCE-side "
            "detector (Part 2) sidesteps the problem entirely.\n"
        )
    else:
        print(
            "**500 noise is not dominated by recognizable citation lead "
            "phrases.** A lead-phrase filter would leave most designator-bearing "
            "500 notes in place, so 500 is hard to salvage by that rule alone. "
            "This strengthens the case for the CCE-side detector (Part 2) over "
            "consuming MARC 500.\n"
        )


def _print_part2_coverage(result: StudyResult) -> None:
    """Emit the Part-2 coverage tables."""
    cov = result.coverage
    print("## Part 2 — CCE part detector: coverage\n")
    print(
        f"Resolved {result.resolved_tagged} of {result.tagged_count} tagged "
        f"whole/part cases ({result.missing_in_index} CCE missing in index).\n"
    )
    print("| category | cases | flagged | coverage |")
    print("|:---|---:|---:|---:|")
    for category in (_CAT_MARC_WHOLE, _CAT_CCE_WHOLE):
        total = cov.by_category_total[category]
        hit = cov.by_category_hit[category]
        rate = f"{hit / total:.0%}" if total else "—"
        print(f"| `{category}` | {total} | {hit} | {rate} |")
    total_all = sum(cov.by_category_total.values())
    hit_all = sum(cov.by_category_hit.values())
    rate_all = f"{hit_all / total_all:.0%}" if total_all else "—"
    print(f"| **overall** | **{total_all}** | **{hit_all}** | **{rate_all}** |")
    print()

    print("### Coverage by firing pattern\n")
    print("| pattern | hits |")
    print("|:---|---:|")
    for pattern, count in cov.pattern_counts.most_common():
        print(f"| `{pattern}` | {count} |")
    print()

    print("### Coverage by CCE source field\n")
    print("| source field | hits |")
    print("|:---|---:|")
    for src in (_SRC_TITLE, _SRC_NOTES, _SRC_DESC):
        print(f"| `{src}` | {cov.field_counts[src]} |")
    print()

    _print_coverage_misses(cov)


def _print_coverage_misses(cov: CoverageReport) -> None:
    """Emit the whole/part cases the detector MISSED (recall gaps)."""
    misses = [case for case in cov.cases if not case.detection.hit]
    print(f"### Misses ({len(misses)} cases the detector did NOT flag)\n")
    if not misses:
        print("_The detector flagged every resolved whole/part case._\n")
        return
    print("| marc_control_id | nypl_uuid | category |")
    print("|:---|:---|:---|")
    for case in misses[:_MAX_REPRESENTATIVE_ROWS]:
        uuid = case.nypl_uuid[:_UUID_TRUNCATE]
        print(f"| {case.marc_control_id} | {uuid}… | {case.category} |")
    print()


def _print_part2_fp(result: StudyResult) -> None:
    """Emit the Part-2 false-positive section."""
    fp = result.fp
    print("## Part 2 — CCE part detector: false-positive rate\n")
    rate = fp.hits / fp.sample_size if fp.sample_size else 0.0
    print(
        f"- Clean-negative pool: {fp.pool_size} entries (verdict "
        f"`{_VERDICT_MATCH}`, untagged, keyword-free note).\n"
        f"- Seeded sample: {fp.sample_size} "
        f"(`Random({_FP_SAMPLE_SEED})`).\n"
        f"- False positives: **{fp.hits}** → **FP rate {rate:.1%}**.\n"
    )
    if fp.pattern_counts:
        print("### False positives by pattern\n")
        print("| pattern | hits |")
        print("|:---|---:|")
        for pattern, count in fp.pattern_counts.most_common():
            print(f"| `{pattern}` | {count} |")
        print()
        print("### False positives by CCE source field\n")
        print("| source field | hits |")
        print("|:---|---:|")
        for src in (_SRC_TITLE, _SRC_NOTES, _SRC_DESC):
            print(f"| `{src}` | {fp.field_counts[src]} |")
        print()
    print("### False-positive examples (what tripped the regex)\n")
    if not fp.examples:
        print("_No false positives in this run._\n")
        return
    print("| marc_control_id | nypl_uuid | pattern | field | matched text |")
    print("|:---|:---|:---|:---|:---|")
    for case in fp.examples:
        uuid = case.nypl_uuid[:_UUID_TRUNCATE]
        det = case.detection
        print(
            f"| {case.marc_control_id} | {uuid}… | {det.pattern} | "
            f"{det.source_field} | {_truncate(det.matched_text, _RAW_TEXT_TRUNCATE)} |"
        )
    print()


def _print_decision(result: StudyResult) -> None:
    """Emit the ship/no-ship decision for the CCE-side detector."""
    cov = result.coverage
    fp = result.fp
    total_all = sum(cov.by_category_total.values())
    hit_all = sum(cov.by_category_hit.values())
    coverage_rate = hit_all / total_all if total_all else 0.0
    fp_rate = fp.hits / fp.sample_size if fp.sample_size else 0.0

    print("## Decision — is the CCE-side detector Phase-1's primary signal?\n")
    print(
        f"- **Coverage**: {coverage_rate:.0%} of resolved whole/part cases "
        f"flagged ({hit_all}/{total_all}).\n"
        f"- **False-positive rate**: {fp_rate:.1%} ({fp.hits}/{fp.sample_size} "
        "clean negatives).\n"
    )
    strong = coverage_rate >= 0.5
    clean = fp_rate <= 0.05
    if strong and clean:
        print(
            "**Recommendation: adopt the CCE-side detector as Phase 1's primary "
            "whole/part signal.** Coverage is high enough to flag the majority of "
            "labeled whole/part cases and the false-positive rate on clean "
            "single-work matches is within tolerance. Consume CCE `title` "
            "(highest-precision), then `notes`, then `desc`, using "
            "`_detect_part` plus the additive series-context patterns above. "
            "Wire it into `volume.py`'s `_classify_cce` so a CCE that looks like "
            "a part is classified `part` from these fields, then proceed to "
            "Phase 2's retrain + held-out eval + `pdm run regression` gate.\n"
        )
    elif strong and not clean:
        print(
            f"**Recommendation: promising coverage but FP rate ({fp_rate:.1%}) is "
            "above the 5% tolerance — tighten before shipping.** Inspect the "
            "false-positive examples above and demote or constrain the offending "
            "pattern(s) (the trailing-comma and parenthetical-series patterns are "
            "the usual precision risks). Re-measure FP after tightening; do NOT "
            "ship the detector as the primary signal until FP is in tolerance.\n"
        )
    else:
        print(
            f"**Recommendation: coverage ({coverage_rate:.0%}) is too low to make "
            "the CCE-side detector the primary signal on its own.** It may still "
            "serve as one feature among several, but the whole/part losses are "
            "not dominated by CCE-side part designators. Reassess whether a "
            "combined MARC+CCE approach is needed.\n"
        )
    print(
        "> Per the vault blind-spot finding, the labeled vault is structurally "
        "biased toward `match` pairs the matcher already surfaces; coverage here "
        "is an UPPER BOUND on top-1 flips, not a prediction. Phase 2's per-MARC "
        "diagnostic remains the real gate.\n"
    )


def _print_report(result: StudyResult) -> None:
    """Emit the full markdown report to stdout."""
    print("# CCE-side part-of-a-larger-work signal — 2026-06-13\n")
    _print_limit_warning(result.limit)
    print(
        "Issue #82, Phase-1 prep. Sizes a CCE-side part detector (the "
        "maintainer's redirected approach) and confirms where the MARC parsed-"
        "notes designator noise comes from by tag. No `src/` code is modified; "
        "the vault is read-only.\n"
    )
    _print_method(result)
    _print_part1(result)
    _print_part2_coverage(result)
    _print_part2_fp(result)
    _print_decision(result)


def _parse_limit() -> int | None:
    """Parse the optional ``--limit N`` smoke flag from argv."""
    parser = ArgumentParser(
        description="CCE part-signal sizing (issue #82, Phase-1 prep)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="cap tagged cases and FP sample (smoke; report is flagged)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    limit: int | None = args.limit
    return limit


def main() -> None:
    """Run both analyses and print the markdown report to stdout."""
    limit = _parse_limit()
    _progress(f"cce_part_signal started (limit={limit})")
    entries = current_entries(_VAULT_PATH)
    result = run_study(entries, limit)
    _print_report(result)
    _progress("report written")


if __name__ == "__main__":
    main()
