"""Measure how many more CCE renewals would join a registration under regnum
normalization of the join key.

Read-only diagnostic. Parses the registration XML tree and the renewal TSV
tree with the SAME streaming iterators the index builder uses
(:func:`iter_nypl_reg_directory`, :func:`iter_nypl_ren_directory`), then
reproduces the builder's exact registration-side join (a sanity gate that
MUST land on the known ``renewal_joins`` figure) before re-counting joins
under four key schemes:

1. baseline  -- raw ``make_renewal_key`` on both sides (no normalization).
2. norm regnum + full ISO date.
3. norm regnum + year only (rescues date-granularity mismatches).
4. norm regnum alone (also rescues renewals whose original date is absent),
   plus a collision audit so we know whether dropping the date risks
   merging distinct registrations that happen to share a normalized regnum.

The script never writes to the corpus or the index; it only reads source
files and prints a numbers-first report.
"""

from collections.abc import Iterable
from datetime import date
from pathlib import Path
from re import Pattern
from re import compile as re_compile
from sys import stdout
from time import perf_counter

from pd_matcher.index.codec import make_renewal_key
from pd_matcher.models import NyplRegRecord
from pd_matcher.models import NyplRenRecord
from pd_matcher.parsers.nypl_reg import iter_nypl_reg_directory
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_directory

_REG_DIR = Path("data/nypl-reg/xml")
_REN_DIR = Path("data/nypl-ren/data")

_EXPECTED_REG_CENTRIC_JOINS = 160_239
_GATE_TOLERANCE = 500

_FOREIGN: Pattern[str] = re_compile(r"^A[\s—–\-]*(?:FOREIGN|FOR)\.?(?=\s|$)")
_INTERIM: Pattern[str] = re_compile(r"^A[\s—–\-]*(?:AD[\s—–\-]*INT|INT)\.?(?=\s|$)")
_NON_ALNUM: Pattern[str] = re_compile(r"[^A-Z0-9]")
_INTERNAL_WS: Pattern[str] = re_compile(r"\S+\s+\S+")
_RANGE_TOKEN: Pattern[str] = re_compile(r"^[A-Z]*[0-9]+$")


def normalize_regnum(raw: str) -> str:
    """Collapse documented CCE registration-number format variance to a canon.

    The transform is applied identically to both the registration ``regnum``
    attribute and the renewal ``oreg`` value, so any deterministic mapping
    that sends the documented variants to one canonical token recovers the
    join. Steps, in order:

    1. Uppercase and strip surrounding whitespace.
    2. Collapse the verbose foreign/interim class phrases the registration
       guide enumerates -- ``A--Foreign 32851`` / ``A for. 48359`` -> ``AF``,
       ``A ad int. 8956`` / ``A int. 241`` -> ``AI`` -- using a leading,
       token-anchored match so a serial like ``A INTERNATIONAL`` is left
       alone.
    3. Drop every remaining non-alphanumeric byte (interior spaces such as
       ``A 963122`` -> ``A963122``, hyphens such as ``AI-9217`` -> ``AI9217``
       and ``B5-73742`` -> ``B573742``, periods, em/en dashes, commas).

    Letter ``O`` and digit ``0`` are intentionally preserved as distinct.

    Args:
        raw: The registration number as transcribed (``regnum`` or ``oreg``).

    Returns:
        The canonical alphanumeric registration-number token (possibly empty
        if ``raw`` held no alphanumerics).
    """
    upper = raw.upper().strip()
    upper = _FOREIGN.sub("AF", upper)
    upper = _INTERIM.sub("AI", upper)
    return _NON_ALNUM.sub("", upper)


def _is_multi_regnum(raw: str) -> bool:
    """Return ``True`` when ``raw`` looks like a space-separated regnum range.

    The guide documents entries whose ``regnum`` attribute carries several
    numbers (``A160078 A160079 A160080``). The single-valued normalizer
    concatenates these into one unmatchable token, so we count them
    separately as a residual lever the normalization schemes do not capture.
    A verbose class phrase (``A ad int. 8956``) also has interior whitespace
    but is not a range, so every whitespace token must independently look
    like a registration number.
    """
    if _INTERNAL_WS.search(raw) is None:
        return False
    tokens = raw.upper().split()
    return len(tokens) > 1 and all(_RANGE_TOKEN.match(token) is not None for token in tokens)


class _RenewalSummary:
    """Renewal-side fields and counters collected in one streaming pass."""

    __slots__ = ("total", "no_oreg", "oreg_no_odat", "records", "baseline_keys")

    def __init__(self) -> None:
        self.total: int = 0
        self.no_oreg: int = 0
        self.oreg_no_odat: int = 0
        self.records: list[tuple[str, date | None]] = []
        self.baseline_keys: set[bytes] = set()


def _collect_renewals(records: Iterable[NyplRenRecord]) -> _RenewalSummary:
    """Stream renewals once, recording join fields and the baseline key set."""
    summary = _RenewalSummary()
    for record in records:
        summary.total += 1
        if record.oreg is None:
            summary.no_oreg += 1
            continue
        if record.odat is None:
            summary.oreg_no_odat += 1
        summary.records.append((record.oreg, record.odat))
        if record.odat is not None:
            summary.baseline_keys.add(make_renewal_key(record.oreg, record.odat))
    return summary


class _RegistrationSets:
    """Registration-side key sets and the normalized-regnum collision audit."""

    __slots__ = (
        "raw",
        "norm_full",
        "norm_year",
        "norm_alone",
        "norm_record_count",
        "norm_first_date",
        "norm_ambiguous",
        "multi_regnum",
        "reg_centric_joins",
        "with_regnum",
    )

    def __init__(self) -> None:
        self.raw: set[bytes] = set()
        self.norm_full: set[str] = set()
        self.norm_year: set[str] = set()
        self.norm_alone: set[str] = set()
        self.norm_record_count: dict[str, int] = {}
        self.norm_first_date: dict[str, str] = {}
        self.norm_ambiguous: set[str] = set()
        self.multi_regnum: int = 0
        self.reg_centric_joins: int = 0
        self.with_regnum: int = 0


def _date_identity(reg_date: date | None, reg_year: int | None) -> str:
    """Return a stable per-registration date token for collision detection."""
    if reg_date is not None:
        return reg_date.isoformat()
    if reg_year is not None:
        return str(reg_year)
    return ""


def _reg_year_for_join(record: NyplRegRecord) -> int | None:
    """Year used by the year-only scheme: registration date first, then fallback."""
    if record.reg_date is not None:
        return record.reg_date.year
    return record.reg_year


def _collect_registrations(
    records: Iterable[NyplRegRecord], renewal_baseline_keys: set[bytes]
) -> _RegistrationSets:
    """Stream registrations once, building every scheme's key set and the audit."""
    sets = _RegistrationSets()
    for record in records:
        if record.regnum is None:
            continue
        sets.with_regnum += 1

        raw_key = make_renewal_key(record.regnum, record.reg_date)
        sets.raw.add(raw_key)
        if raw_key in renewal_baseline_keys:
            sets.reg_centric_joins += 1

        if _is_multi_regnum(record.regnum):
            sets.multi_regnum += 1

        norm = normalize_regnum(record.regnum)
        if not norm:
            continue

        iso = record.reg_date.isoformat() if record.reg_date is not None else ""
        sets.norm_full.add(f"{norm}|{iso}")
        year = _reg_year_for_join(record)
        sets.norm_year.add(f"{norm}|{year if year is not None else ''}")
        sets.norm_alone.add(norm)

        sets.norm_record_count[norm] = sets.norm_record_count.get(norm, 0) + 1
        identity = _date_identity(record.reg_date, record.reg_year)
        seen = sets.norm_first_date.get(norm)
        if seen is None:
            sets.norm_first_date[norm] = identity
        elif seen != identity:
            sets.norm_ambiguous.add(norm)
    return sets


class _SchemeCounts:
    """Renewal-centric join counts under each key scheme."""

    __slots__ = (
        "baseline",
        "norm_full",
        "norm_year",
        "norm_alone",
        "alone_only_gain",
        "alone_ambiguous",
        "alone_only_ambiguous",
    )

    def __init__(self) -> None:
        self.baseline: int = 0
        self.norm_full: int = 0
        self.norm_year: int = 0
        self.norm_alone: int = 0
        self.alone_only_gain: int = 0
        self.alone_ambiguous: int = 0
        self.alone_only_ambiguous: int = 0


def _count_scheme_joins(summary: _RenewalSummary, sets: _RegistrationSets) -> _SchemeCounts:
    """Count, per scheme, how many renewals match a registration key."""
    counts = _SchemeCounts()
    for oreg, odat in summary.records:
        norm = normalize_regnum(oreg)
        matched_year = False
        if odat is not None:
            if make_renewal_key(oreg, odat) in sets.raw:
                counts.baseline += 1
            if f"{norm}|{odat.isoformat()}" in sets.norm_full:
                counts.norm_full += 1
            if norm and f"{norm}|{odat.year}" in sets.norm_year:
                counts.norm_year += 1
                matched_year = True
        if norm and norm in sets.norm_alone:
            counts.norm_alone += 1
            ambiguous = norm in sets.norm_ambiguous
            if ambiguous:
                counts.alone_ambiguous += 1
            if not matched_year:
                counts.alone_only_gain += 1
                if ambiguous:
                    counts.alone_only_ambiguous += 1
    return counts


def _print_report(summary: _RenewalSummary, sets: _RegistrationSets, counts: _SchemeCounts) -> None:
    """Emit the numbers-first recovery report."""
    write = stdout.write
    base = counts.baseline
    shared = sum(1 for value in sets.norm_record_count.values() if value > 1)
    max_norm = max(sets.norm_record_count, key=sets.norm_record_count.get, default="")
    max_share = sets.norm_record_count.get(max_norm, 0)

    write("\n=== Renewal join recovery under regnum normalization ===\n\n")
    write(f"registrations parsed (with regnum): {sets.with_regnum:,}\n")
    write(f"total renewals parsed:              {summary.total:,}\n")
    write(f"  renewals with no oreg:            {summary.no_oreg:,}\n")
    write(f"  renewals with oreg but no odat:   {summary.oreg_no_odat:,}\n\n")

    write(f"SANITY GATE (reg-centric, replicates builder): {sets.reg_centric_joins:,}\n")
    write(f"  expected renewal_joins:                      {_EXPECTED_REG_CENTRIC_JOINS:,}\n")
    delta_gate = sets.reg_centric_joins - _EXPECTED_REG_CENTRIC_JOINS
    gate_ok = abs(delta_gate) <= _GATE_TOLERANCE
    write(f"  delta vs expected:                           {delta_gate:+,}  ")
    write("PASS\n\n" if gate_ok else "FAIL\n\n")

    write("scheme                         renewals joined     delta vs baseline\n")
    write("-" * 70 + "\n")
    rows = (
        ("1 baseline (raw regnum+date)", counts.baseline),
        ("2 norm regnum + full date", counts.norm_full),
        ("3 norm regnum + year only", counts.norm_year),
        ("4 norm regnum alone", counts.norm_alone),
    )
    for label, value in rows:
        write(f"{label:<30} {value:>15,}     {value - base:>+15,}\n")

    write("\n--- regnum-alone collision audit ---\n")
    write(f"distinct normalized regnums:                 {len(sets.norm_alone):,}\n")
    write(f"normalized regnums on >1 registration:       {shared:,}\n")
    write(f"normalized regnums on >1 distinct date:      {len(sets.norm_ambiguous):,}\n")
    write(f"max registrations sharing one norm regnum:   {max_share:,}  ({max_norm})\n")
    write(f"renewals joined by scheme 4 only (date dropped): {counts.alone_only_gain:,}\n")
    write(f"  of those landing on an ambiguous regnum:       {counts.alone_only_ambiguous:,}\n")
    write(f"all scheme-4 joins landing on ambiguous regnum:  {counts.alone_ambiguous:,}\n")

    write("\n--- residual lever (not captured by normalization) ---\n")
    write(f"reg regnums that are space-separated ranges:  {sets.multi_regnum:,}\n")
    stdout.flush()


def main() -> None:
    """Run the full read-only measurement and print the report."""
    start = perf_counter()
    write = stdout.write
    write("parsing renewals ...\n")
    stdout.flush()
    summary = _collect_renewals(iter_nypl_ren_directory(_REN_DIR))
    write(f"  {summary.total:,} renewals in {perf_counter() - start:,.1f}s\n")
    stdout.flush()

    reg_start = perf_counter()
    write("parsing registrations (this is the slow pass) ...\n")
    stdout.flush()
    sets = _collect_registrations(iter_nypl_reg_directory(_REG_DIR), summary.baseline_keys)
    write(f"  {sets.with_regnum:,} registrations in {perf_counter() - reg_start:,.1f}s\n")
    stdout.flush()

    counts = _count_scheme_joins(summary, sets)
    _print_report(summary, sets, counts)
    write(f"\ntotal runtime: {perf_counter() - start:,.1f}s\n")
    stdout.flush()


if __name__ == "__main__":
    main()
