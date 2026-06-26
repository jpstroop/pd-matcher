"""Characterize space-separated multi-number CCE registration ``regnum`` values.

Read-only diagnostic. A subset of CCE registrations carry several numbers in
the single ``regnum`` attribute (``"A160078 A160079 A160080"``). The merged
single-valued :func:`normalize_regnum` concatenates these into one
unmatchable token, so today they neither join renewals nor expose a
whole/part signal to the matcher. This script answers two questions, using
the SAME streaming iterators the index builder uses
(:func:`iter_nypl_reg_directory`, :func:`iter_nypl_ren_directory`):

1. Are these registered multi-volume WHOLES? -- population shape (count,
   numbers-per-record distribution, consecutive-run fraction) plus a
   volume-signal characterization that runs the production ``volume.compat``
   detectors over each record's own title/desc/notes/edition.
2. What would per-number expansion buy for renewal joins? -- indexing such a
   registration under EACH number, then counting how many renewals whose
   ``oreg`` cites one of the interior/any numbers would ADDITIONALLY join
   beyond the single-valued normalization already merged.

The script never writes to the corpus or the index; it only reads source
files and prints a numbers-first report.
"""

from collections.abc import Iterable
from datetime import date
from random import Random
from re import Pattern
from re import compile as re_compile
from statistics import median
from sys import stdout
from time import perf_counter

from pd_matcher.index.codec import make_renewal_key
from pd_matcher.models import NyplRegRecord
from pd_matcher.models import NyplRenRecord
from pd_matcher.normalize.registration_numbers import normalize_regnum
from pd_matcher.parsers.nypl_reg import iter_nypl_reg_directory
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_directory
from pd_matcher.match.scorers.volume import _detect_bare_designator
from pd_matcher.match.scorers.volume import _detect_part
from pd_matcher.match.scorers.volume import _is_multivolume_whole
from pd_matcher.match.scorers.volume import _is_part_range
from pathlib import Path

_REG_DIR = Path("data/nypl-reg/xml")
_REN_DIR = Path("data/nypl-ren/data")

_SAMPLE_SEED = 8224
_SAMPLE_SIZE = 30
_TITLE_EXAMPLE_COUNT = 10

_INTERNAL_WS: Pattern[str] = re_compile(r"\S+\s+\S+")
_RANGE_TOKEN: Pattern[str] = re_compile(r"^[A-Z]*[0-9]+$")
_PREFIX_NUMBER: Pattern[str] = re_compile(r"^([A-Z]*)([0-9]+)$")


def _is_multi_regnum(raw: str) -> bool:
    """Return ``True`` when ``raw`` looks like a space-separated regnum range.

    Mirrors the helper prototyped in ``scripts/renewal_join_recovery.py``: a
    verbose class phrase (``"A ad int. 8956"``) also has interior whitespace
    but is not a range, so every whitespace token must independently look
    like a registration number.
    """
    if _INTERNAL_WS.search(raw) is None:
        return False
    tokens = raw.upper().split()
    return len(tokens) > 1 and all(_RANGE_TOKEN.match(token) is not None for token in tokens)


def _regnum_numbers(raw: str) -> list[str]:
    """Return the per-number normalized tokens of a multi-number ``regnum``."""
    return [normalize_regnum(token) for token in raw.split() if normalize_regnum(token)]


def _is_consecutive_run(tokens: list[str]) -> bool:
    """Return ``True`` when ``tokens`` are one prefix with +1 ascending numbers."""
    parsed: list[tuple[str, int]] = []
    for token in tokens:
        match = _PREFIX_NUMBER.match(token)
        if match is None:
            return False
        parsed.append((match.group(1), int(match.group(2))))
    if len({prefix for prefix, _ in parsed}) != 1:
        return False
    numbers = [number for _, number in parsed]
    return all(numbers[index + 1] == numbers[index] + 1 for index in range(len(numbers) - 1))


def _looks_volume_ish(values: Iterable[str | None]) -> tuple[bool, bool, bool]:
    """Return ``(any_signal, whole_or_range, single_part)`` over the CCE fields.

    ``whole_or_range`` fires on a covering designator range (``"Vol. 1-2"``)
    or a multi-volume whole count (``"3 v."``) -- the CCE-whole direction the
    ``volume.compat`` scorer reads. ``single_part`` fires on a lone part
    designator. ``any_signal`` is the union, including bare Roman/digit
    subtitle designators.
    """
    whole_or_range = False
    single_part = False
    any_other = False
    for value in values:
        if not value:
            continue
        if _is_part_range(value) or _is_multivolume_whole(value):
            whole_or_range = True
        if _detect_part(value) is not None:
            single_part = True
        if _detect_bare_designator(value) is not None:
            any_other = True
    return (whole_or_range or single_part or any_other, whole_or_range, single_part)


class _MultiRegSample:
    """One sampled multi-number registration, kept for characterization."""

    __slots__ = ("regnum", "count", "consecutive", "title", "desc", "notes", "edition")

    def __init__(self, record: NyplRegRecord, numbers: list[str]) -> None:
        self.regnum: str = record.regnum or ""
        self.count: int = len(numbers)
        self.consecutive: bool = _is_consecutive_run(numbers)
        self.title: str = record.title
        self.desc: str | None = record.desc
        self.notes: tuple[str, ...] = record.notes
        self.edition: str | None = record.edition

    def fields(self) -> tuple[str | None, ...]:
        """Return the CCE text fields scanned for a volume signal."""
        return (self.title, self.desc, self.edition, *self.notes)


class _RenewalSide:
    """Renewal join fields collected in one streaming pass."""

    __slots__ = ("total", "no_oreg", "multi_oreg", "records")

    def __init__(self) -> None:
        self.total: int = 0
        self.no_oreg: int = 0
        self.multi_oreg: int = 0
        self.records: list[tuple[str, date | None]] = []


def _collect_renewals(records: Iterable[NyplRenRecord]) -> _RenewalSide:
    """Stream renewals once, recording (normalized oreg, odat) join fields."""
    side = _RenewalSide()
    for record in records:
        side.total += 1
        if record.oreg is None:
            side.no_oreg += 1
            continue
        if _is_multi_regnum(record.oreg):
            side.multi_oreg += 1
        side.records.append((record.oreg, record.odat))
    return side


class _RegistrationSide:
    """Registration-side key sets, multi-regnum stats, and the sample reservoir."""

    __slots__ = (
        "with_regnum",
        "multi_count",
        "multi_with_date",
        "consecutive",
        "number_counts",
        "current_full",
        "current_year",
        "current_alone",
        "exp_full_first",
        "exp_full_interior",
        "exp_year_first",
        "exp_year_interior",
        "exp_alone_first",
        "exp_alone_interior",
        "vol_any",
        "vol_whole",
        "vol_part",
        "samples",
        "examples",
        "_rng",
        "_seen_multi",
    )

    def __init__(self) -> None:
        self.with_regnum: int = 0
        self.multi_count: int = 0
        self.multi_with_date: int = 0
        self.consecutive: int = 0
        self.number_counts: list[int] = []
        self.current_full: set[str] = set()
        self.current_year: set[str] = set()
        self.current_alone: set[str] = set()
        self.exp_full_first: set[str] = set()
        self.exp_full_interior: set[str] = set()
        self.exp_year_first: set[str] = set()
        self.exp_year_interior: set[str] = set()
        self.exp_alone_first: set[str] = set()
        self.exp_alone_interior: set[str] = set()
        self.vol_any: int = 0
        self.vol_whole: int = 0
        self.vol_part: int = 0
        self.samples: list[_MultiRegSample] = []
        self.examples: list[_MultiRegSample] = []
        self._rng = Random(_SAMPLE_SEED)
        self._seen_multi: int = 0


def _reg_year(record: NyplRegRecord) -> int | None:
    """Year used by the year-relaxed scheme: registration date first, fallback."""
    if record.reg_date is not None:
        return record.reg_date.year
    return record.reg_year


def _add_current_keys(side: _RegistrationSide, norm: str, record: NyplRegRecord) -> None:
    """Record the production (current, whole-string) join keys for ``record``."""
    iso = record.reg_date.isoformat() if record.reg_date is not None else ""
    year = _reg_year(record)
    side.current_full.add(f"{norm}|{iso}")
    side.current_year.add(f"{norm}|{year if year is not None else ''}")
    side.current_alone.add(norm)


def _add_expansion_keys(side: _RegistrationSide, numbers: list[str], record: NyplRegRecord) -> None:
    """Record per-number expansion keys, split first-number vs interior."""
    iso = record.reg_date.isoformat() if record.reg_date is not None else ""
    year = _reg_year(record)
    year_token = str(year) if year is not None else ""
    for index, number in enumerate(numbers):
        full = f"{number}|{iso}"
        yr = f"{number}|{year_token}"
        if index == 0:
            side.exp_full_first.add(full)
            side.exp_year_first.add(yr)
            side.exp_alone_first.add(number)
        else:
            side.exp_full_interior.add(full)
            side.exp_year_interior.add(yr)
            side.exp_alone_interior.add(number)


def _reservoir_sample(side: _RegistrationSide, sample: _MultiRegSample) -> None:
    """Maintain a uniform reservoir of ``_SAMPLE_SIZE`` multi-regnum records."""
    side._seen_multi += 1
    if len(side.samples) < _SAMPLE_SIZE:
        side.samples.append(sample)
        return
    slot = side._rng.randint(0, side._seen_multi - 1)
    if slot < _SAMPLE_SIZE:
        side.samples[slot] = sample


def _collect_registrations(records: Iterable[NyplRegRecord]) -> _RegistrationSide:
    """Stream registrations once, building key sets, stats, and the sample."""
    side = _RegistrationSide()
    for record in records:
        if record.regnum is None:
            continue
        side.with_regnum += 1
        norm = normalize_regnum(record.regnum)
        if norm:
            _add_current_keys(side, norm, record)
        if not _is_multi_regnum(record.regnum):
            continue
        numbers = _regnum_numbers(record.regnum)
        if len(numbers) < 2:
            continue
        side.multi_count += 1
        side.number_counts.append(len(numbers))
        if record.reg_date is not None:
            side.multi_with_date += 1
        if _is_consecutive_run(numbers):
            side.consecutive += 1
        sample = _MultiRegSample(record, numbers)
        any_sig, whole_sig, part_sig = _looks_volume_ish(sample.fields())
        if any_sig:
            side.vol_any += 1
        if whole_sig:
            side.vol_whole += 1
        if part_sig:
            side.vol_part += 1
        if any_sig and len(side.examples) < _TITLE_EXAMPLE_COUNT:
            side.examples.append(sample)
        _add_expansion_keys(side, numbers, record)
        _reservoir_sample(side, sample)
    return side


class _ExpansionGain:
    """Additional renewal joins per-number expansion buys, per date scheme."""

    __slots__ = ("full_any", "full_interior", "year_any", "year_interior", "alone_any",
                 "alone_interior")

    def __init__(self) -> None:
        self.full_any: int = 0
        self.full_interior: int = 0
        self.year_any: int = 0
        self.year_interior: int = 0
        self.alone_any: int = 0
        self.alone_interior: int = 0


def _count_expansion_gain(renewals: _RenewalSide, side: _RegistrationSide) -> _ExpansionGain:
    """Count renewals that join ONLY via a per-number expansion key."""
    gain = _ExpansionGain()
    for oreg, odat in renewals.records:
        norm = normalize_regnum(oreg)
        if not norm:
            continue
        iso = odat.isoformat() if odat is not None else ""
        year = str(odat.year) if odat is not None else ""
        full = f"{norm}|{iso}"
        if full not in side.current_full:
            if full in side.exp_full_interior:
                gain.full_any += 1
                gain.full_interior += 1
            elif full in side.exp_full_first:
                gain.full_any += 1
        yr = f"{norm}|{year}"
        if yr not in side.current_year:
            if yr in side.exp_year_interior:
                gain.year_any += 1
                gain.year_interior += 1
            elif yr in side.exp_year_first:
                gain.year_any += 1
        if norm not in side.current_alone:
            if norm in side.exp_alone_interior:
                gain.alone_any += 1
                gain.alone_interior += 1
            elif norm in side.exp_alone_first:
                gain.alone_any += 1
    return gain


def _print_population(side: _RegistrationSide) -> None:
    """Emit the multi-regnum population shape."""
    write = stdout.write
    counts = side.number_counts
    write("\n=== Range-regnum characterization ===\n\n")
    write("--- population ---\n")
    write(f"registrations parsed (with regnum):        {side.with_regnum:,}\n")
    write(f"multi-number regnum records:               {side.multi_count:,}\n")
    write(f"  with a reg_date (full-date join usable):  {side.multi_with_date:,}\n")
    if counts:
        write(f"numbers per record: median {median(counts):.1f}  ")
        write(f"mean {sum(counts) / len(counts):.2f}  min {min(counts)}  max {max(counts)}\n")
        histogram: dict[str, int] = {}
        for value in counts:
            bucket = str(value) if value <= 9 else "10+"
            histogram[bucket] = histogram.get(bucket, 0) + 1
        ordered = sorted(histogram, key=lambda key: (key == "10+", int(key.rstrip("+"))))
        write("  distribution: " + "  ".join(f"{key}={histogram[key]:,}" for key in ordered) + "\n")
    consec_pct = 100.0 * side.consecutive / side.multi_count if side.multi_count else 0.0
    arbitrary = side.multi_count - side.consecutive
    write(f"strictly CONSECUTIVE runs (e.g. A160078-80): {side.consecutive:,}  ({consec_pct:.1f}%)\n")
    write(f"arbitrary lists / gaps / mixed prefix:       {arbitrary:,}\n")


def _print_volume_signal(side: _RegistrationSide) -> None:
    """Emit the volume-signal characterization and concrete title examples."""
    write = stdout.write
    total = side.multi_count or 1
    write("\n--- multi-volume signal (production volume.compat detectors) ---\n")
    write(f"ANY volume indicator (title/desc/notes/edition): {side.vol_any:,}  "
          f"({100.0 * side.vol_any / total:.1f}%)\n")
    write(f"  whole / covering-range indicator:               {side.vol_whole:,}  "
          f"({100.0 * side.vol_whole / total:.1f}%)\n")
    write(f"  single part-designator indicator:               {side.vol_part:,}  "
          f"({100.0 * side.vol_part / total:.1f}%)\n")

    write(f"\nconcrete title examples (first {len(side.examples)} with a volume signal):\n")
    for sample in side.examples:
        write(f"  regnum={sample.regnum!r}  n={sample.count}  consec={sample.consecutive}\n")
        write(f"    title: {sample.title!r}\n")
        if sample.desc:
            write(f"    desc:  {sample.desc!r}\n")
        for note in sample.notes[:2]:
            write(f"    note:  {note!r}\n")

    write(f"\nuniform random sample ({len(side.samples)} of {side.multi_count:,}, seed {_SAMPLE_SEED}):\n")
    for sample in side.samples:
        any_sig, _whole, _part = _looks_volume_ish(sample.fields())
        flag = "VOL" if any_sig else "   "
        write(f"  [{flag}] regnum={sample.regnum!r}  n={sample.count}  "
              f"consec={sample.consecutive}  title={sample.title!r}\n")


def _print_expansion(renewals: _RenewalSide, gain: _ExpansionGain) -> None:
    """Emit the renewal cross-reference and per-number expansion gain."""
    write = stdout.write
    write("\n--- renewal cross-ref (additional joins from per-number expansion) ---\n")
    write(f"total renewals parsed:              {renewals.total:,}\n")
    write(f"  renewals with no oreg:            {renewals.no_oreg:,}\n")
    write(f"  renewals with multi-number oreg:  {renewals.multi_oreg:,}\n\n")
    write("additional renewal joins beyond the merged single-valued normalize:\n")
    write("scheme                       any number of a range     interior (NOT first) only\n")
    write("-" * 84 + "\n")
    write(f"full date (production-faithful) {gain.full_any:>14,}     {gain.full_interior:>20,}\n")
    write(f"year only (date relaxed)       {gain.year_any:>14,}     {gain.year_interior:>20,}\n")
    write(f"regnum alone (date dropped)    {gain.alone_any:>14,}     {gain.alone_interior:>20,}\n")
    stdout.flush()


def main() -> None:
    """Run the full read-only measurement and print the report."""
    start = perf_counter()
    write = stdout.write
    write("parsing renewals ...\n")
    stdout.flush()
    renewals = _collect_renewals(iter_nypl_ren_directory(_REN_DIR))
    write(f"  {renewals.total:,} renewals in {perf_counter() - start:,.1f}s\n")
    stdout.flush()

    reg_start = perf_counter()
    write("parsing registrations (slow pass) ...\n")
    stdout.flush()
    side = _collect_registrations(iter_nypl_reg_directory(_REG_DIR))
    write(f"  {side.with_regnum:,} registrations in {perf_counter() - reg_start:,.1f}s\n")
    stdout.flush()

    gain = _count_expansion_gain(renewals, side)
    _print_population(side)
    _print_volume_signal(side)
    _print_expansion(renewals, gain)
    write(f"\ntotal runtime: {perf_counter() - start:,.1f}s\n")
    stdout.flush()


if __name__ == "__main__":
    main()
