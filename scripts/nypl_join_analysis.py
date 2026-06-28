"""Definitive cardinality analysis of the NYPL registration<->renewal join.

Read-only decision proof. Parses the FULL CCE registration XML tree and the
FULL renewal TSV tree with the SAME streaming iterators the index builder
uses (:func:`iter_nypl_reg_directory`, :func:`iter_nypl_ren_directory`) and
builds join keys with the SAME production functions
(:func:`make_renewal_keys`, which internally calls
:func:`normalize_regnum` / :func:`is_multi_regnum`). Nothing is sampled and
nothing is reimplemented, so every number reflects exactly what the index
does.

The question that gates the "four-scenario" renewal pipeline: does a
renewal's ``(normalized oreg + odat)`` point to AT MOST ONE registration in
the overwhelming majority of cases? If many-to-one joins are common the
"reg-match within the odat year -> one registration" assumption breaks.

Two join granularities are measured side by side because the plan wording is
ambiguous between them:

* EXACT-DATE -- the production key ``normalize_regnum(regnum)|isoformat(date)``
  (reg side keyed on the strict ``reg_date``, renewal side on ``odat``). This
  is literally what ``ren_by_oreg`` stores.
* YEAR-LEVEL -- ``(normalized regnum, year)`` (reg side keyed on the derived
  ``reg_year``, renewal side on ``odat.year``). This is the looser
  "within the odat year" reading and rescues date-granularity mismatches.

The script never writes to the corpus or the index; it only reads source
files and prints a numbers-first report to stdout.
"""

from collections import Counter
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from sys import argv
from sys import stdout
from time import perf_counter

from pd_matcher.index.codec import make_renewal_keys
from pd_matcher.models import NyplRegRecord
from pd_matcher.models import NyplRenRecord
from pd_matcher.normalize.registration_numbers import is_multi_regnum
from pd_matcher.normalize.registration_numbers import normalize_regnum
from pd_matcher.parsers.nypl_reg import iter_nypl_reg_directory
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_directory

_REG_DIR = Path("data/nypl-reg/xml")
_REN_DIR = Path("data/nypl-ren/data")

_EXPECTED_REGS = 2_168_402
_EXPECTED_RENS = 443_693
_EXPECTED_JOINS = 167_317

_TITLE_TRUNC = 90
_EXAMPLE_CAP = 60


def _log(message: str) -> None:
    """Write a timestamped milestone to the shared agent progress log."""
    with Path("/tmp/agent-progress.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{perf_counter():.0f}s join-analysis: {message}\n")


def regnum_tokens(regnum: str) -> tuple[str, ...]:
    """Return the normalized regnum token(s), mirroring :func:`make_renewal_keys`.

    A single value collapses to one ``normalize_regnum`` token; a recognised
    space-separated multi-number range fans out into one normalized token per
    listed number, exactly as the codec builds keys (minus the date suffix).
    """
    if not is_multi_regnum(regnum):
        return (normalize_regnum(regnum),)
    return tuple(normalize_regnum(token) for token in regnum.split())


def _trunc(value: str | None) -> str:
    """Return ``value`` truncated for single-line display, or a placeholder."""
    if value is None:
        return "-"
    collapsed = " ".join(value.split())
    if len(collapsed) <= _TITLE_TRUNC:
        return collapsed
    return collapsed[: _TITLE_TRUNC - 1] + "…"


def _percentile(sorted_values: list[int], fraction: float) -> int:
    """Return the ``fraction`` percentile of an ascending list (nearest-rank)."""
    if not sorted_values:
        return 0
    index = int(fraction * (len(sorted_values) - 1))
    return sorted_values[index]


def _bucketize(counts: Iterable[int]) -> Counter[str]:
    """Bucket per-key cardinalities into human-readable size ranges."""
    buckets: Counter[str] = Counter()
    for count in counts:
        if count <= 5:
            buckets[str(count)] += 1
        elif count <= 10:
            buckets["6-10"] += 1
        elif count <= 50:
            buckets["11-50"] += 1
        elif count <= 100:
            buckets["51-100"] += 1
        else:
            buckets["101+"] += 1
    return buckets


class _RenewalCorpus:
    """All renewal-side data needed for the join analysis, loaded once."""

    __slots__ = (
        "author",
        "joinable",
        "key_to_ridx",
        "no_odat",
        "no_oreg",
        "no_oreg_and_odat",
        "odat",
        "odat_year_counter",
        "oreg",
        "rdat_year_counter",
        "title",
        "total",
    )

    def __init__(self) -> None:
        self.oreg: list[str | None] = []
        self.odat: list[date | None] = []
        self.title: list[str | None] = []
        self.author: list[str | None] = []
        self.key_to_ridx: dict[bytes, list[int]] = {}
        self.odat_year_counter: Counter[int] = Counter()
        self.rdat_year_counter: Counter[int] = Counter()
        self.total = 0
        self.no_oreg = 0
        self.no_odat = 0
        self.no_oreg_and_odat = 0
        self.joinable = 0


def _load_renewals(records: Iterable[NyplRenRecord]) -> _RenewalCorpus:
    """Stream every renewal once, recording fields, stats, and the join keys."""
    corpus = _RenewalCorpus()
    append_oreg = corpus.oreg.append
    append_odat = corpus.odat.append
    append_title = corpus.title.append
    append_author = corpus.author.append
    for record in records:
        ridx = corpus.total
        corpus.total += 1
        append_oreg(record.oreg)
        append_odat(record.odat)
        append_title(record.title)
        append_author(record.author)
        if record.oreg is None:
            corpus.no_oreg += 1
        if record.odat is None:
            corpus.no_odat += 1
        else:
            corpus.odat_year_counter[record.odat.year] += 1
        if record.rdat is not None:
            corpus.rdat_year_counter[record.rdat.year] += 1
        if record.oreg is None or record.odat is None:
            corpus.no_oreg_and_odat += 1
            continue
        corpus.joinable += 1
        for key in make_renewal_keys(record.oreg, record.odat):
            corpus.key_to_ridx.setdefault(key, []).append(ridx)
        if corpus.total % 100_000 == 0:
            _log(f"renewals loaded {corpus.total:,}")
    return corpus


class _RegistrationCorpus:
    """All registration-side data needed for the join analysis, loaded once."""

    __slots__ = (
        "joins",
        "key_to_ridx",
        "multi_regnum",
        "no_reg_date",
        "no_regnum",
        "regnum",
        "regnum_year_counts",
        "renewals_per_reg",
        "reg_date",
        "reg_year",
        "reg_year_counter",
        "title",
        "total",
        "uuid",
        "with_reg_date",
        "with_regnum",
    )

    def __init__(self) -> None:
        self.uuid: list[str] = []
        self.title: list[str] = []
        self.regnum: list[str | None] = []
        self.reg_date: list[str] = []
        self.reg_year: list[int] = []
        self.key_to_ridx: dict[bytes, list[int]] = {}
        self.regnum_year_counts: dict[str, dict[int, int]] = {}
        self.reg_year_counter: Counter[int] = Counter()
        self.renewals_per_reg: Counter[int] = Counter()
        self.total = 0
        self.with_regnum = 0
        self.no_regnum = 0
        self.with_reg_date = 0
        self.no_reg_date = 0
        self.multi_regnum = 0
        self.joins = 0


def _load_registrations(
    records: Iterable[NyplRegRecord], ren_keys: dict[bytes, list[int]]
) -> _RegistrationCorpus:
    """Stream every registration once.

    Builds the reg-side key multimap, the (regnum, year) counts, the year
    histogram, and the per-registration renewal fan-out (which reproduces the
    builder's ``renewal_joins`` figure: a registration counts as renewed when
    at least one of its keys hits the renewal key map).
    """
    corpus = _RegistrationCorpus()
    for record in records:
        ridx = corpus.total
        corpus.total += 1
        corpus.uuid.append(record.uuid)
        corpus.title.append(_trunc(record.title))
        corpus.regnum.append(record.regnum)
        corpus.reg_date.append(record.reg_date.isoformat() if record.reg_date is not None else "")
        corpus.reg_year.append(record.reg_year if record.reg_year is not None else -1)
        if record.reg_year is not None:
            corpus.reg_year_counter[record.reg_year] += 1
        if record.reg_date is not None:
            corpus.with_reg_date += 1
        else:
            corpus.no_reg_date += 1

        if record.regnum is None:
            corpus.no_regnum += 1
            corpus.renewals_per_reg[0] += 1
            continue
        corpus.with_regnum += 1
        if is_multi_regnum(record.regnum):
            corpus.multi_regnum += 1

        keys = make_renewal_keys(record.regnum, record.reg_date)
        for key in keys:
            corpus.key_to_ridx.setdefault(key, []).append(ridx)

        if record.reg_year is not None:
            for token in regnum_tokens(record.regnum):
                if not token:
                    continue
                year_counts = corpus.regnum_year_counts.setdefault(token, {})
                year_counts[record.reg_year] = year_counts.get(record.reg_year, 0) + 1

        matched: set[int] = set()
        for key in keys:
            renewal_ridxs = ren_keys.get(key)
            if renewal_ridxs is not None:
                matched.update(renewal_ridxs)
        corpus.renewals_per_reg[len(matched)] += 1
        if matched:
            corpus.joins += 1
        if corpus.total % 200_000 == 0:
            _log(f"registrations loaded {corpus.total:,}")
    return corpus


class _RenewalFanout:
    """Per-renewal fan-out distributions (exact-date and year-level)."""

    __slots__ = (
        "exact_dist",
        "exact_many_examples",
        "unjoinable",
        "year_dist",
        "year_many_examples",
    )

    def __init__(self) -> None:
        self.exact_dist: Counter[int] = Counter()
        self.year_dist: Counter[int] = Counter()
        self.unjoinable = 0
        self.exact_many_examples: list[tuple[int, list[int]]] = []
        self.year_many_examples: list[tuple[int, str, int]] = []


def _compute_renewal_fanout(rens: _RenewalCorpus, regs: _RegistrationCorpus) -> _RenewalFanout:
    """For every renewal, count distinct registrations its keys match.

    ``exact_dist`` keys on the production date-precise join; ``year_dist`` on
    the looser ``(normalized regnum, year)`` join. ``unjoinable`` collects
    renewals lacking ``oreg`` or ``odat`` (no key, cannot join at all).
    """
    fanout = _RenewalFanout()
    reg_keys = regs.key_to_ridx
    regnum_year = regs.regnum_year_counts
    for ridx in range(rens.total):
        oreg = rens.oreg[ridx]
        odat = rens.odat[ridx]
        if oreg is None or odat is None:
            fanout.unjoinable += 1
            continue
        keys = make_renewal_keys(oreg, odat)
        matched: set[int] = set()
        for key in keys:
            reg_ridxs = reg_keys.get(key)
            if reg_ridxs is not None:
                matched.update(reg_ridxs)
        exact = len(matched)
        fanout.exact_dist[exact] += 1
        if exact >= 2 and len(fanout.exact_many_examples) < _EXAMPLE_CAP:
            fanout.exact_many_examples.append((ridx, sorted(matched)))

        year = odat.year
        year_total = 0
        for token in regnum_tokens(oreg):
            if not token:
                continue
            year_total += regnum_year.get(token, {}).get(year, 0)
        fanout.year_dist[year_total] += 1
        if year_total >= 2 and len(fanout.year_many_examples) < _EXAMPLE_CAP:
            primary = regnum_tokens(oreg)[0]
            fanout.year_many_examples.append((ridx, primary, year))
    return fanout


def _write_year_histogram(label: str, counter: Counter[int]) -> None:
    """Print an ascending year histogram with a coverage span summary."""
    write = stdout.write
    if not counter:
        write(f"{label}: (none)\n")
        return
    years = sorted(counter)
    write(f"{label} span {years[0]}-{years[-1]}, distinct years {len(years)}\n")
    for year in years:
        write(f"    {year}  {counter[year]:>9,}\n")


def _write_cardinality(label: str, key_to_ridx: dict[bytes, list[int]]) -> None:
    """Print distinct-key, collision, and per-key size statistics for a multimap."""
    write = stdout.write
    sizes = sorted(len(ridxs) for ridxs in key_to_ridx.values())
    distinct = len(sizes)
    collisions = sum(1 for size in sizes if size > 1)
    shared_members = sum(size for size in sizes if size > 1)
    write(f"\n--- {label} key cardinality ---\n")
    write(f"distinct keys:                       {distinct:,}\n")
    write(f"keys mapping to >1 record:           {collisions:,}")
    if distinct:
        write(f"  ({collisions / distinct:.4%} of keys)\n")
    else:
        write("\n")
    write(f"records sharing a key with another:  {shared_members:,}\n")
    write(f"max records per key:                 {sizes[-1] if sizes else 0:,}\n")
    write(f"p99 records per key:                 {_percentile(sizes, 0.99)}\n")
    write(f"p999 records per key:                {_percentile(sizes, 0.999)}\n")
    write("size distribution (records-per-key -> #keys):\n")
    for bucket, count in sorted(_bucketize(sizes).items(), key=_bucket_sort_key):
        write(f"    {bucket:>6}  {count:>9,}\n")


def _bucket_sort_key(item: tuple[str, int]) -> tuple[int, int]:
    """Sort bucket labels numerically (single ints before ranges)."""
    label = item[0]
    if label.isdigit():
        return (0, int(label))
    return (1, int(label.split("-")[0].rstrip("+")))


def _write_fanout(label: str, dist: Counter[int], total: int) -> None:
    """Print a fan-out distribution (registrations matched per renewal)."""
    write = stdout.write
    write(f"\n--- {label} ---\n")
    zero = dist.get(0, 0)
    one = dist.get(1, 0)
    many = sum(count for value, count in dist.items() if value >= 2)
    joined = one + many
    write(f"renewals matching 0 registrations:   {zero:,}\n")
    write(f"renewals matching exactly 1:         {one:,}\n")
    write(f"renewals matching >1 (many-to-one):  {many:,}\n")
    if total:
        write(f"  many-to-one as % of all renewals:  {many / total:.4%}\n")
    if joined:
        write(f"  many-to-one as % of joined renewals: {many / joined:.4%}\n")
    write("matched-count distribution (regs-per-renewal -> #renewals):\n")
    for value in sorted(dist):
        write(f"    {value:>4}  {dist[value]:>9,}\n")


def _write_renewals_per_reg(regs: _RegistrationCorpus) -> None:
    """Print the symmetric distribution: renewals matched per registration."""
    write = stdout.write
    dist = regs.renewals_per_reg
    write("\n--- registration -> renewal fan-out (symmetric) ---\n")
    with_one = dist.get(1, 0)
    with_many = sum(count for value, count in dist.items() if value >= 2)
    write(f"registrations with 0 renewals:       {dist.get(0, 0):,}\n")
    write(f"registrations with exactly 1:        {with_one:,}\n")
    write(f"registrations with >1 renewal:       {with_many:,}\n")
    write(f"registrations with >=1 (= joins):    {with_one + with_many:,}\n")
    write("renewals-per-registration distribution:\n")
    for value in sorted(dist):
        if value <= 10 or value % 5 == 0:
            write(f"    {value:>4}  {dist[value]:>9,}\n")


def _write_regnum_year_audit(regs: _RegistrationCorpus) -> None:
    """Print serial-reuse and (regnum, year) uniqueness statistics."""
    write = stdout.write
    distinct_regnums = len(regs.regnum_year_counts)
    multi_year = 0
    year_dup_groups = 0
    year_dup_members = 0
    max_years = 0
    max_years_regnum = ""
    for regnum, year_counts in regs.regnum_year_counts.items():
        if len(year_counts) > 1:
            multi_year += 1
            if len(year_counts) > max_years:
                max_years = len(year_counts)
                max_years_regnum = regnum
        for count in year_counts.values():
            if count > 1:
                year_dup_groups += 1
                year_dup_members += count
    write("\n--- regnum-within-year uniqueness + serial reuse ---\n")
    write(f"distinct normalized regnums (with reg_year):     {distinct_regnums:,}\n")
    write(f"regnums appearing in >1 distinct year (reuse):   {multi_year:,}")
    if distinct_regnums:
        write(f"  ({multi_year / distinct_regnums:.4%})\n")
    else:
        write("\n")
    write(f"max distinct years for one regnum:               {max_years}  ({max_years_regnum})\n")
    write(f"(regnum, year) groups with >1 registration:      {year_dup_groups:,}\n")
    write(f"registrations in such within-year-dup groups:    {year_dup_members:,}\n")


def _write_exact_examples(fanout: _RenewalFanout, rens: _RenewalCorpus, regs: _RegistrationCorpus) -> None:
    """Print concrete renewals whose EXACT-DATE key matched >1 registration."""
    write = stdout.write
    write("\n=== EXACT-DATE many-to-one examples (renewal -> registrations) ===\n")
    shown = 0
    for ridx, reg_ridxs in fanout.exact_many_examples:
        if shown >= 10:
            break
        shown += 1
        oreg = rens.oreg[ridx]
        odat = rens.odat[ridx]
        odat_str = odat.isoformat() if odat is not None else "-"
        write(f"\nrenewal entry: oreg={oreg!r} odat={odat_str}\n")
        write(f"  title:  {_trunc(rens.title[ridx])}\n")
        write(f"  author: {_trunc(rens.author[ridx])}\n")
        write(f"  -> {len(reg_ridxs)} registrations share this exact key:\n")
        for reg_ridx in reg_ridxs[:6]:
            write(
                f"     reg uuid={regs.uuid[reg_ridx]} regnum={regs.regnum[reg_ridx]!r} "
                f"reg_date={regs.reg_date[reg_ridx] or '-'} year={regs.reg_year[reg_ridx]}\n"
            )
            write(f"        title: {regs.title[reg_ridx]}\n")


def _write_year_examples(fanout: _RenewalFanout, rens: _RenewalCorpus, regs: _RegistrationCorpus) -> None:
    """Print concrete renewals whose YEAR-LEVEL key matched >1 registration.

    The matching registrations are resolved by an in-memory scan over the
    already-parsed registration arrays (no re-parse), so the example shows the
    actual colliding titles/years for the selected ``(regnum, year)`` pairs.
    """
    write = stdout.write
    write("\n=== YEAR-LEVEL many-to-one examples (renewal -> registrations) ===\n")
    wanted: dict[tuple[str, int], list[int]] = {}
    selected = fanout.year_many_examples[:10]
    for _ridx, regnum, year in selected:
        wanted[(regnum, year)] = []
    for reg_ridx in range(regs.total):
        regnum_raw = regs.regnum[reg_ridx]
        if regnum_raw is None:
            continue
        year = regs.reg_year[reg_ridx]
        if year < 0:
            continue
        for token in regnum_tokens(regnum_raw):
            bucket = wanted.get((token, year))
            if bucket is not None and len(bucket) < 6:
                bucket.append(reg_ridx)
    for ridx, regnum, year in selected:
        oreg = rens.oreg[ridx]
        odat = rens.odat[ridx]
        odat_str = odat.isoformat() if odat is not None else "-"
        members = wanted.get((regnum, year), [])
        write(f"\nrenewal entry: oreg={oreg!r} odat={odat_str} (norm regnum={regnum} year={year})\n")
        write(f"  title:  {_trunc(rens.title[ridx])}\n")
        write(f"  -> registrations sharing (regnum, year):\n")
        for reg_ridx in members:
            write(
                f"     reg uuid={regs.uuid[reg_ridx]} regnum={regs.regnum[reg_ridx]!r} "
                f"reg_date={regs.reg_date[reg_ridx] or '-'} year={regs.reg_year[reg_ridx]}\n"
            )
            write(f"        title: {regs.title[reg_ridx]}\n")


def _write_high_renewal_regs(regs: _RegistrationCorpus) -> None:
    """Print registrations carrying many renewals (symmetric extreme cases)."""
    write = stdout.write
    write("\n=== registrations matched by many renewals (top of distribution) ===\n")
    top = sorted(regs.renewals_per_reg, reverse=True)[:5]
    for value in top:
        write(f"  {regs.renewals_per_reg[value]:,} registrations matched {value} renewals each\n")


def _write_header(rens: _RenewalCorpus, regs: _RegistrationCorpus) -> None:
    """Print the corpus/coverage section and reconciliation against the index."""
    write = stdout.write
    write("\n=== NYPL registration<->renewal join: definitive cardinality ===\n")
    write("\n--- corpus + reconciliation ---\n")
    write(f"registrations parsed:   {regs.total:,}  (index meta {_EXPECTED_REGS:,})\n")
    write(f"renewals parsed:        {rens.total:,}  (index meta {_EXPECTED_RENS:,})\n")
    write(f"  with regnum:          {regs.with_regnum:,}\n")
    write(f"  no regnum:            {regs.no_regnum:,}\n")
    write(f"  with strict reg_date: {regs.with_reg_date:,}\n")
    write(f"  no strict reg_date:   {regs.no_reg_date:,}  (cannot exact-join a renewal)\n")
    write(f"  multi-number regnums: {regs.multi_regnum:,}\n")
    write(f"renewal_joins (reg-centric): {regs.joins:,}  (index meta {_EXPECTED_JOINS:,}, "
          f"delta {regs.joins - _EXPECTED_JOINS:+,})\n")
    write("\n--- renewal join-field coverage ---\n")
    write(f"renewals with no oreg:        {rens.no_oreg:,}\n")
    write(f"renewals with no odat:        {rens.no_odat:,}  (missing/unparseable -> cannot join)\n")
    write(f"renewals lacking oreg or odat:{rens.no_oreg_and_odat:,}\n")
    write(f"joinable renewals (both set): {rens.joinable:,}\n")


def main() -> None:
    """Run the full read-only measurement over the entire corpus."""
    reg_dir = Path(argv[1]) if len(argv) > 1 else _REG_DIR
    ren_dir = Path(argv[2]) if len(argv) > 2 else _REN_DIR
    start = perf_counter()
    write = stdout.write

    _log("start; parsing renewals")
    write("parsing renewals ...\n")
    stdout.flush()
    rens = _load_renewals(iter_nypl_ren_directory(ren_dir))
    write(f"  {rens.total:,} renewals in {perf_counter() - start:,.1f}s\n")
    _log(f"renewals done {rens.total:,}; parsing registrations (slow)")
    stdout.flush()

    reg_start = perf_counter()
    write("parsing registrations (slow pass) ...\n")
    stdout.flush()
    regs = _load_registrations(iter_nypl_reg_directory(reg_dir), rens.key_to_ridx)
    write(f"  {regs.total:,} registrations in {perf_counter() - reg_start:,.1f}s\n")
    _log(f"registrations done {regs.total:,}; computing fan-out")
    stdout.flush()

    fanout = _compute_renewal_fanout(rens, regs)
    _log("fan-out computed; writing report")

    _write_header(rens, regs)
    write("\n--- registration reg_year histogram ---\n")
    _write_year_histogram("reg_year", regs.reg_year_counter)
    write("\n--- renewal odat-year histogram ---\n")
    _write_year_histogram("odat year", rens.odat_year_counter)
    write("\n--- renewal rdat-year histogram ---\n")
    _write_year_histogram("rdat year", rens.rdat_year_counter)

    _write_cardinality("REGISTRATION (exact-date)", regs.key_to_ridx)
    _write_cardinality("RENEWAL (exact-date)", rens.key_to_ridx)

    _write_fanout(
        "EXACT-DATE renewal fan-out (regs per renewal)", fanout.exact_dist, rens.joinable
    )
    _write_fanout(
        "YEAR-LEVEL renewal fan-out (regs per renewal)", fanout.year_dist, rens.joinable
    )
    write(f"\nrenewals unjoinable (no oreg/odat): {fanout.unjoinable:,}\n")

    _write_renewals_per_reg(regs)
    _write_regnum_year_audit(regs)
    _write_exact_examples(fanout, rens, regs)
    _write_year_examples(fanout, rens, regs)
    _write_high_renewal_regs(regs)

    write(f"\ntotal runtime: {perf_counter() - start:,.1f}s\n")
    stdout.flush()
    _log(f"DONE total {perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
