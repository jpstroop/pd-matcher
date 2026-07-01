"""Definitive full-corpus measurement of the ``<additionalEntry>`` join yield.

Read-only decision proof gating issue #111. The production registration
parser (:mod:`pd_matcher.parsers.nypl_reg`) iterates only
``<copyrightEntry>`` and keeps a single top-level ``regnum``; the CCE guide
("Multiple claims in a single entry") documents that one ``<copyrightEntry>``
can bundle several separate registrations as ``<additionalEntry>`` children,
each carrying its own ``<regNum>`` and ``<regDate>``. Those interior numbers
are dropped today, so a renewal that cites an interior registration cannot
join.

This script answers, over the WHOLE corpus with no sampling, how many
additional reg<->renewal joins indexing the ``<additionalEntry>`` keys would
recover, and how book-relevant that yield is. Every join key is built with
the SAME production function the index uses (:func:`make_renewal_keys`, which
internally calls :func:`normalize_regnum` / :func:`is_multi_regnum`), so the
keys are byte-identical to what the index stores. The top-level registration
side reuses the production private helpers (:func:`_extract_reg_date`,
:func:`_text`) and applies the same id/title guards as
:func:`pd_matcher.parsers.nypl_reg._build_record`, so the reg-centric join
count reconciles with the index's ``renewal_joins`` meta value as a validity
check.

Nothing is written to the corpus, the index, or any file under ``data/``.
The script only reads source files and prints a numbers-first report to
stdout.
"""

from collections import Counter
from datetime import date
from pathlib import Path
from re import Pattern
from re import compile as re_compile
from sqlite3 import connect
from sys import argv
from sys import stdout
from time import perf_counter

from lxml.etree import _Element
from lxml.etree import iterparse

from pd_matcher.index.codec import make_renewal_keys
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.parsers.nypl_reg import NyplRegParseStats
from pd_matcher.parsers.nypl_reg import _extract_reg_date
from pd_matcher.parsers.nypl_reg import _text
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_directory

_REG_DIR = Path("data/nypl-reg/xml")
_REN_DIR = Path("data/nypl-ren/data")
_INDEX_PATH = Path("caches/cce.lmdb")

_RENEWAL_REVIEW_DB = Path("data/renewal_review.db")
_REVIEW_DB = Path("data/review.db")

_ENTRY_TAG = "copyrightEntry"
_ADDITIONAL_TAG = "additionalEntry"
_RENEWAL_ENTRY_TAG = "renewalEntry"

_CLASS_PREFIX: Pattern[str] = re_compile(r"^([A-Z]+)")


def _log(message: str) -> None:
    """Write a timestamped milestone to the shared agent progress log."""
    with Path("/tmp/agent-progress.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{perf_counter():.0f}s addl-entry: {message}\n")


def _regnum_class(normalized: str) -> str:
    """Infer the CCE registration class from a normalized regnum's prefix.

    Class is the leading run of letters, with the periodical-contribution
    forms ``A5``/``B5`` (normalized ``A5CT``/``A5NT`` -> ``A5``) split out
    from bare ``A``/``B`` because the guide flags them as distinct. Anything
    without a leading letter (a bare numeric class) collapses to ``num``.
    This is INFORMATIONAL only — never a scope filter (a book MARC can match
    a periodical-class renewal; see docs/COPYRIGHT_SCENARIOS.md, pair 429).
    """
    match = _CLASS_PREFIX.match(normalized)
    if match is None:
        return "num" if normalized else "empty"
    letters = match.group(1)
    rest = normalized[match.end() :]
    if letters in ("A", "B") and rest.startswith("5"):
        return f"{letters}5"
    return letters


def _key_class(key: bytes) -> str:
    """Return the registration class inferred from a ``regnum|date`` join key."""
    regnum = key.split(b"|", 1)[0].decode("ascii", "replace")
    return _regnum_class(regnum)


class _RenewalSide:
    """Renewal-side join keys and per-renewal identity, loaded once."""

    __slots__ = (
        "entry_and_id_to_ridx",
        "joinable",
        "keys",
        "no_odat",
        "no_oreg",
        "ren_id_set",
        "ren_key_set",
        "total",
    )

    def __init__(self) -> None:
        self.keys: list[tuple[bytes, ...]] = []
        self.ren_key_set: set[bytes] = set()
        self.ren_id_set: set[str] = set()
        self.entry_and_id_to_ridx: dict[tuple[str, str], int] = {}
        self.total = 0
        self.joinable = 0
        self.no_oreg = 0
        self.no_odat = 0


def _load_renewals(ren_dir: Path) -> _RenewalSide:
    """Stream every renewal once, recording production join keys and identity.

    ``keys[ridx]`` holds the tuple of ``make_renewal_keys(oreg, odat)`` for a
    joinable renewal (empty tuple when it lacks ``oreg`` or ``odat``).
    ``entry_and_id_to_ridx`` maps ``(entry_id, renewal id)`` to the row index
    so the scenario-4 review rows can be resolved to the exact renewal that
    produced them (an ``entry_id`` can carry several renewals).
    """
    side = _RenewalSide()
    for record in iter_nypl_ren_directory(ren_dir):
        ridx = side.total
        side.total += 1
        side.entry_and_id_to_ridx[(record.entry_id, record.id)] = ridx
        side.ren_id_set.add(record.id)
        if record.oreg is None:
            side.no_oreg += 1
        if record.odat is None:
            side.no_odat += 1
        if record.oreg is None or record.odat is None:
            side.keys.append(())
            continue
        side.joinable += 1
        keys = make_renewal_keys(record.oreg, record.odat)
        side.keys.append(keys)
        side.ren_key_set.update(keys)
        if side.total % 100_000 == 0:
            _log(f"renewals loaded {side.total:,}")
    return side


class _RegSide:
    """Registration-side aggregates from one custom iterparse pass."""

    __slots__ = (
        "add_entries",
        "add_entries_no_own_date",
        "add_entries_no_regnum",
        "add_entries_with_regnum",
        "add_key_class",
        "add_key_fallback",
        "add_key_strict",
        "entries_skipped_no_id_or_title",
        "regs_joined_top",
        "regs_total",
        "regs_with_add",
        "top_key_set",
    )

    def __init__(self) -> None:
        self.top_key_set: set[bytes] = set()
        self.add_key_strict: set[bytes] = set()
        self.add_key_fallback: set[bytes] = set()
        self.add_key_class: dict[bytes, set[str]] = {}
        self.regs_total = 0
        self.regs_joined_top = 0
        self.regs_with_add = 0
        self.entries_skipped_no_id_or_title = 0
        self.add_entries = 0
        self.add_entries_with_regnum = 0
        self.add_entries_no_regnum = 0
        self.add_entries_no_own_date = 0


def _clear_element(elem: _Element) -> None:
    """Clear a finished element and drop already-processed previous siblings."""
    elem.clear()
    previous = elem.getprevious()
    while previous is not None:
        parent = elem.getparent()
        if parent is None:  # pragma: no cover - top-level guard
            break
        del parent[0]
        previous = elem.getprevious()


def _harvest_additional_entries(
    entry: _Element,
    parent_reg_date: date | None,
    side: _RegSide,
    stats: NyplRegParseStats,
) -> None:
    """Fan every ``<additionalEntry>`` child into the additional-key sets.

    The additionalEntry's own ``regnum`` attribute is preferred over its
    inline ``<regNum>`` text (mirroring the top-level convention). Two key
    sets are built to bracket the yield:

    * STRICT — the additionalEntry's own ``<regDate>`` only (``None`` when
      absent). This mirrors production's strict :func:`_extract_reg_date` join
      key exactly; an absent date yields an empty-suffix key that can never
      collide with a dated renewal, so those entries simply contribute nothing.
      This is the trustworthy floor.
    * FALLBACK — own ``<regDate>`` if present, else the PARENT entry's regDate
      (as issue #111 requested). The parent date is usually a DIFFERENT
      registration event, so most fallback-only joins are coincidental; this
      is an upper bound, not a trustworthy count.

    Keys are built with the production :func:`make_renewal_keys` so they are
    byte-identical to what the index would store.
    """
    for add in entry.iterfind(_ADDITIONAL_TAG):
        side.add_entries += 1
        regnum = add.get("regnum") or _text(add.find("regNum"), stats)
        if regnum is None:
            side.add_entries_no_regnum += 1
            continue
        side.add_entries_with_regnum += 1
        own_date = _extract_reg_date(add)
        fallback_date = own_date if own_date is not None else parent_reg_date
        if own_date is None:
            side.add_entries_no_own_date += 1
        else:
            for key in make_renewal_keys(regnum, own_date):
                side.add_key_strict.add(key)
                side.add_key_class.setdefault(key, set()).add(_key_class(key))
        for key in make_renewal_keys(regnum, fallback_date):
            side.add_key_fallback.add(key)
            side.add_key_class.setdefault(key, set()).add(_key_class(key))


def _load_registrations(reg_dir: Path, ren_key_set: set[bytes]) -> _RegSide:
    """Single custom iterparse pass over the whole registration XML tree.

    Reproduces the production top-level extraction (id/title guards, ``regnum``
    attribute preferred over inline ``<regNum>`` text, strict ``<regDate>``)
    to build the top-level key set and the reg-centric join count, and in the
    same pass harvests every ``<additionalEntry>``.
    """
    side = _RegSide()
    stats = NyplRegParseStats()
    for xml_path in sorted(reg_dir.rglob("*.xml")):
        context = iterparse(str(xml_path), events=("end",), tag=_ENTRY_TAG)
        for _event, entry in context:
            uuid = entry.get("id")
            title = _text(entry.find("title"), stats)
            if uuid is None or not uuid.strip() or title is None:
                side.entries_skipped_no_id_or_title += 1
                _clear_element(entry)
                continue

            side.regs_total += 1
            regnum = entry.get("regnum") or _text(entry.find("regNum"), stats)
            reg_date = _extract_reg_date(entry)
            if regnum is not None:
                keys = make_renewal_keys(regnum, reg_date)
                side.top_key_set.update(keys)
                if any(key in ren_key_set for key in keys):
                    side.regs_joined_top += 1

            if entry.find(_ADDITIONAL_TAG) is not None:
                side.regs_with_add += 1
                _harvest_additional_entries(entry, reg_date, side, stats)

            _clear_element(entry)
        if side.regs_total and side.regs_total % 400_000 < 5000:
            _log(f"registrations processed ~{side.regs_total:,}")
        del context
    return side


class _RenewalEntryScan:
    """Counts for the standalone ``<renewalEntry>`` blocks in the reg volumes."""

    __slots__ = ("cites_known_registration", "novel_renewal_num", "total")

    def __init__(self) -> None:
        self.total = 0
        self.cites_known_registration = 0
        self.novel_renewal_num = 0


def _scan_renewal_entries(
    reg_dir: Path, top_key_set: set[bytes], ren_id_set: set[str]
) -> _RenewalEntryScan:
    """Scan the standalone ``<renewalEntry>`` blocks (Q4).

    A ``<renewalEntry>`` is a renewal transcribed directly in a registration
    volume (a top-level sibling, NOT a ``<copyrightEntry>`` child). It cites its
    original registration under ``renewal/registrations/registration``
    (``<regDate>`` + ``<regNum>``) and carries its own ``<renewalNum>``. Two
    contributions are counted: ``cites_known_registration`` — the cited
    registration key hits a top-level registration we hold (so that registration
    would be marked renewed) — and ``novel_renewal_num`` — the ``<renewalNum>``
    is NOT already present in the renewal TSV corpus (a genuinely new renewal).
    """
    scan = _RenewalEntryScan()
    stats = NyplRegParseStats()
    for xml_path in sorted(reg_dir.rglob("*.xml")):
        context = iterparse(str(xml_path), events=("end",), tag=_RENEWAL_ENTRY_TAG)
        for _event, ren in context:
            scan.total += 1
            renewal = ren.find("renewal")
            if renewal is None:
                _clear_element(ren)
                continue
            renewal_num = _text(renewal.find("renewalNum"), stats)
            if renewal_num is not None and renewal_num not in ren_id_set:
                scan.novel_renewal_num += 1
            registrations = renewal.find("registrations")
            hit = False
            if registrations is not None:
                for registration in registrations.iterfind("registration"):
                    regnum = _text(registration.find("regNum"), stats)
                    if regnum is None:
                        continue
                    reg_date = _extract_reg_date(registration)
                    if any(key in top_key_set for key in make_renewal_keys(regnum, reg_date)):
                        hit = True
                        break
            if hit:
                scan.cites_known_registration += 1
            _clear_element(ren)
        del context
    return scan


class _RenewalJoinOutcome:
    """Per-renewal join outcomes under top-level vs additionalEntry keys.

    ``net_new_strict`` counts renewals joining NO top-level key but joining an
    own-regDate additionalEntry key (the trustworthy floor). ``net_new_fallback``
    is the same under the parent-date fallback set (an upper bound).
    ``outcome_by_entry_id`` records ``(joined_top, joined_add_strict,
    joined_add_fallback)`` per renewal row for the scenario-4 pass.
    """

    __slots__ = (
        "class_edges_strict",
        "joined_top",
        "net_new_fallback",
        "net_new_strict",
        "outcome_by_entry_id",
        "unjoined_current",
    )

    def __init__(self) -> None:
        self.joined_top = 0
        self.net_new_strict = 0
        self.net_new_fallback = 0
        self.unjoined_current = 0
        self.class_edges_strict: Counter[str] = Counter()
        self.outcome_by_entry_id: dict[int, tuple[bool, bool, bool]] = {}


def _classify_renewals(rens: _RenewalSide, regs: _RegSide) -> _RenewalJoinOutcome:
    """For every joinable renewal, resolve top-level vs additionalEntry joins.

    The net-new counts capture renewals that join NO top-level registration key
    but DO join an additionalEntry key — the headline recovered joins — under
    both the strict (own-regDate) and fallback (parent-date) additionalEntry key
    sets. For each strict net-new renewal, the joining additionalEntry key(s)
    contribute one edge per key to the class distribution (informational).
    """
    outcome = _RenewalJoinOutcome()
    top = regs.top_key_set
    add_strict = regs.add_key_strict
    add_fallback = regs.add_key_fallback
    add_class = regs.add_key_class
    for ridx in range(rens.total):
        keys = rens.keys[ridx]
        if not keys:
            continue
        joined_top = any(key in top for key in keys)
        joined_strict = any(key in add_strict for key in keys)
        joined_fallback = any(key in add_fallback for key in keys)
        outcome.outcome_by_entry_id[ridx] = (joined_top, joined_strict, joined_fallback)
        if joined_top:
            outcome.joined_top += 1
            continue
        outcome.unjoined_current += 1
        if joined_fallback:
            outcome.net_new_fallback += 1
        if joined_strict:
            outcome.net_new_strict += 1
            for key in keys:
                if key in add_strict:
                    for cls in add_class.get(key, ()):
                        outcome.class_edges_strict[cls] += 1
    return outcome


class _Scenario4Impact:
    """Bogus-scenario-4 tally over the renewal review DB."""

    __slots__ = (
        "already_joined",
        "becomes_joined_fallback",
        "becomes_joined_strict",
        "matched",
        "still_unjoined",
        "total",
        "unresolved",
    )

    def __init__(self) -> None:
        self.total = 0
        self.matched = 0
        self.unresolved = 0
        self.already_joined = 0
        self.becomes_joined_strict = 0
        self.becomes_joined_fallback = 0
        self.still_unjoined = 0


def _measure_scenario4(
    db_path: Path, rens: _RenewalSide, outcome: _RenewalJoinOutcome
) -> _Scenario4Impact:
    """Count scenario-4 review rows that become joined once additionalEntry keys exist.

    Each ``pairing_type='renewal'`` row is resolved to its exact renewal via
    ``(nypl_uuid, cce_renewal_id) == (entry_id, renewal id)``. A row is
    ``becomes_joined`` (bogus) when its renewal does not join any top-level
    key but does join an additionalEntry key, reported under both the strict
    (own-regDate) and fallback (parent-date) additionalEntry key sets.
    """
    impact = _Scenario4Impact()
    connection = connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = connection.execute(
            "SELECT nypl_uuid, cce_renewal_id FROM review_pair WHERE pairing_type = 'renewal'"
        )
        rows = cursor.fetchall()
    finally:
        connection.close()
    for entry_id, renewal_id in rows:
        impact.total += 1
        ridx = rens.entry_and_id_to_ridx.get((entry_id, renewal_id))
        if ridx is None:
            impact.unresolved += 1
            continue
        joined = outcome.outcome_by_entry_id.get(ridx)
        if joined is None:
            impact.unresolved += 1
            continue
        impact.matched += 1
        joined_top, joined_strict, joined_fallback = joined
        if joined_top:
            impact.already_joined += 1
            continue
        if joined_strict:
            impact.becomes_joined_strict += 1
        if joined_fallback:
            impact.becomes_joined_fallback += 1
        if not joined_fallback:
            impact.still_unjoined += 1
    return impact


def _pct(part: int, whole: int) -> str:
    """Return ``part/whole`` as a percentage string, guarding against zero."""
    if whole == 0:
        return "n/a"
    return f"{part / whole:.4%}"


def _write_report(
    rens: _RenewalSide,
    regs: _RegSide,
    outcome: _RenewalJoinOutcome,
    renewal_entries: _RenewalEntryScan,
    index_joins: int,
    index_regs: int,
    index_rens: int,
    scenario4: _Scenario4Impact | None,
    scenario4_db: Path | None,
) -> None:
    """Print the numbers-first report answering every question in issue #111."""
    write = stdout.write

    write("\n=== additionalEntry join-yield measurement (full corpus) ===\n")

    write("\n--- corpus + reconciliation (Q1) ---\n")
    write(f"registrations parsed (guarded):  {regs.regs_total:,}  (index meta {index_regs:,}, "
          f"delta {regs.regs_total - index_regs:+,})\n")
    write(f"entries skipped (no id/title):   {regs.entries_skipped_no_id_or_title:,}\n")
    write(f"renewals parsed:                 {rens.total:,}  (index meta {index_rens:,}, "
          f"delta {rens.total - index_rens:+,})\n")
    write(f"reg-centric top-level joins:     {regs.regs_joined_top:,}  (meta {index_joins:,}, "
          f"d {regs.regs_joined_top - index_joins:+,})\n")
    write(f"distinct top-level reg keys:     {len(regs.top_key_set):,}\n")

    write("\n--- renewal join-field coverage ---\n")
    write(f"renewals with no oreg:           {rens.no_oreg:,}\n")
    write(f"renewals with no odat:           {rens.no_odat:,}\n")
    write(f"joinable renewals (oreg+odat):   {rens.joinable:,}\n")
    write(f"distinct renewal keys:           {len(rens.ren_key_set):,}\n")

    write("\n--- additionalEntry inventory ---\n")
    write(f"copyrightEntry with >=1 addl:    {regs.regs_with_add:,}\n")
    write(f"additionalEntry elements:        {regs.add_entries:,}\n")
    write(f"  with a usable regnum:          {regs.add_entries_with_regnum:,}\n")
    write(f"  no regnum (unusable):          {regs.add_entries_no_regnum:,}\n")
    write(f"  lacking own <regDate>:         {regs.add_entries_no_own_date:,}  "
          f"(no strict key; fallback uses parent regDate)\n")
    write(f"distinct addl keys (strict):     {len(regs.add_key_strict):,}\n")
    write(f"distinct addl keys (fallback):   {len(regs.add_key_fallback):,}\n")
    write(f"  strict keys also top-level:    "
          f"{len(regs.add_key_strict & regs.top_key_set):,}\n")

    write("\n--- additionalEntry yield: NET NEW joins (Q2, headline) ---\n")
    write("STRICT = additionalEntry's own <regDate> only (trustworthy floor; mirrors\n")
    write("production's strict join key). FALLBACK = own date else parent date (upper\n")
    write("bound; parent date is usually a different registration event, so most\n")
    write("fallback-only joins are coincidental).\n")
    write(f"joinable renewals joining top-level:          {outcome.joined_top:,}\n")
    write(f"joinable renewals currently UNJOINED:         {outcome.unjoined_current:,}\n")
    write(f"  NET NEW via addl (STRICT):                  {outcome.net_new_strict:,}\n")
    write(f"  NET NEW via addl (FALLBACK):                {outcome.net_new_fallback:,}\n")
    write(f"  strict net-new as % of currently-unjoined:  "
          f"{_pct(outcome.net_new_strict, outcome.unjoined_current)}\n")
    write(f"  strict net-new as % of all joinable:        "
          f"{_pct(outcome.net_new_strict, rens.joinable)}\n")
    write(f"  strict net-new vs current reg-centric joins:"
          f"{_pct(outcome.net_new_strict, index_joins)}\n")
    write(f"  fallback net-new as % of currently-unjoined:"
          f"{_pct(outcome.net_new_fallback, outcome.unjoined_current)}\n")

    write("\n--- class distribution of STRICT net-new joins (Q3, informational) ---\n")
    write("NOTE: class is NOT a scope filter; a book MARC can match a periodical-class renewal.\n")
    write("(edges = renewal <-> additionalEntry-key joins, counted per joining key)\n")
    book_family = {"A", "AA", "AF", "AI", "AO", "O"}
    total_edges = sum(outcome.class_edges_strict.values())
    book_edges = 0
    for cls, count in sorted(outcome.class_edges_strict.items(), key=lambda kv: (-kv[1], kv[0])):
        marker = "  [book family]" if cls in book_family else ""
        if cls in ("A5", "B5"):
            marker = "  [periodical contribution]"
        if cls in book_family:
            book_edges += count
        write(f"    {cls:>6}  {count:>8,}  ({_pct(count, total_edges)}){marker}\n")
    write(f"  total edges:            {total_edges:,}\n")
    write(f"  book-family edges:      {book_edges:,}  ({_pct(book_edges, total_edges)})\n")

    write("\n--- renewalEntry blocks in reg volumes (Q4, minor) ---\n")
    write("(standalone renewal records embedded in the reg XML, not copyrightEntry children)\n")
    write(f"renewalEntry elements found:     {renewal_entries.total:,}\n")
    write(f"  citing a registration we hold: {renewal_entries.cites_known_registration:,}  "
          f"(would mark that reg renewed)\n")
    write(f"  renewalNum not in renewal TSV: {renewal_entries.novel_renewal_num:,}  "
          f"(genuinely new renewals)\n")

    write("\n--- bogus scenario-4 impact (Q5) ---\n")
    if scenario4 is None or scenario4_db is None:
        write("no renewal review DB found; skipped\n")
    else:
        write(f"review DB:                       {scenario4_db}\n")
        write(f"pairing_type='renewal' rows:     {scenario4.total:,}\n")
        write(f"  resolved to a corpus renewal:  {scenario4.matched:,}\n")
        write(f"  unresolved (no id match):      {scenario4.unresolved:,}\n")
        write(f"  already joined top-level:      {scenario4.already_joined:,}\n")
        write(f"  BECOME joined (STRICT/bogus):  {scenario4.becomes_joined_strict:,}\n")
        write(f"  BECOME joined (FALLBACK):      {scenario4.becomes_joined_fallback:,}\n")
        write(f"  still genuinely unjoined:      {scenario4.still_unjoined:,}\n")
        write(f"  strict bogus as % of resolved: "
              f"{_pct(scenario4.becomes_joined_strict, scenario4.matched)}\n")
        write(f"  strict bogus as % of all rows: "
              f"{_pct(scenario4.becomes_joined_strict, scenario4.total)}\n")
    stdout.flush()


def main() -> None:
    """Run the full read-only measurement over the entire corpus."""
    reg_dir = Path(argv[1]) if len(argv) > 1 else _REG_DIR
    ren_dir = Path(argv[2]) if len(argv) > 2 else _REN_DIR
    start = perf_counter()
    write = stdout.write

    with NyplIndexLookup(_INDEX_PATH) as lookup:
        stats = lookup.stats()
    index_joins = stats.renewal_joins
    index_regs = stats.registrations_written
    index_rens = stats.renewals_written

    _log("start; loading renewals")
    write("loading renewals ...\n")
    stdout.flush()
    rens = _load_renewals(ren_dir)
    write(f"  {rens.total:,} renewals in {perf_counter() - start:,.1f}s\n")
    _log(f"renewals done {rens.total:,}; parsing registrations (slow)")
    stdout.flush()

    reg_start = perf_counter()
    write("parsing registrations + additionalEntry (slow pass) ...\n")
    stdout.flush()
    regs = _load_registrations(reg_dir, rens.ren_key_set)
    write(f"  {regs.regs_total:,} registrations in {perf_counter() - reg_start:,.1f}s\n")
    _log(f"registrations done {regs.regs_total:,}; classifying renewals")
    stdout.flush()

    outcome = _classify_renewals(rens, regs)
    _log("renewals classified; scanning renewalEntry blocks")

    renewal_entries = _scan_renewal_entries(reg_dir, regs.top_key_set, rens.ren_id_set)
    _log("renewalEntry scan done; measuring scenario-4")

    scenario4_db: Path | None = None
    if _RENEWAL_REVIEW_DB.exists():
        scenario4_db = _RENEWAL_REVIEW_DB
    elif _REVIEW_DB.exists():
        scenario4_db = _REVIEW_DB
    scenario4 = (
        _measure_scenario4(scenario4_db, rens, outcome) if scenario4_db is not None else None
    )

    _write_report(
        rens,
        regs,
        outcome,
        renewal_entries,
        index_joins,
        index_regs,
        index_rens,
        scenario4,
        scenario4_db,
    )

    write(f"\ntotal runtime: {perf_counter() - start:,.1f}s\n")
    stdout.flush()
    _log(f"DONE total {perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
