"""Post-rebuild, full-corpus verification of the join-maximization gains.

Read-only decision proof gating issues #108 (year-based join keys), #111
(``<additionalEntry>`` interior join keys), and #113 (harvesting free
MARC<->renewal training positives). The registration index was just rebuilt
with both key families; this script confirms the gains materialized on the
renewal-centric axis and produces the two decision numbers the next phase
depends on.

Everything is built with the CURRENT production surface so the measurement is
byte-identical to what the rebuilt index stores:

* Registrations stream through :func:`iter_nypl_reg_directory`; each
  :class:`NyplRegRecord` now carries ``additional_join_keys``.
* Renewals stream through :func:`iter_nypl_ren_directory`.
* Year-granularity join keys come from the current
  :func:`make_renewal_keys` (``regnum``, ``year:int``) — the same call the
  builder makes in :func:`pd_matcher.index.builder._registration_join_keys`.

The COMPLETE registration keyspace mirrors the builder exactly: every
``make_renewal_keys(regnum, reg_year)`` PLUS every
``make_renewal_keys(additional_regnum, additional_year)`` the parser
harvested from ``<additionalEntry>`` children. The exact-date baseline is
reconstructed locally (``normalize_regnum(regnum)|isoformat(date)``, top-level
only) so the year-vs-additionalEntry gain can be attributed cleanly; the
production codec no longer emits exact-date keys.

The script never writes to the corpus, the index, or any file under
``data/``. It reads source files and the (read-only) review DB and vault, and
prints a numbers-first report to stdout.
"""

from datetime import date
from pathlib import Path
from sqlite3 import connect
from sys import argv
from sys import stdout
from time import perf_counter

from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import iter_entries
from pd_matcher.index.codec import make_renewal_keys
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.models import NyplRegRecord
from pd_matcher.normalize.registration_numbers import is_multi_regnum
from pd_matcher.normalize.registration_numbers import normalize_regnum
from pd_matcher.parsers.nypl_reg import iter_nypl_reg_directory
from pd_matcher.parsers.nypl_ren import iter_nypl_ren_directory

_REG_DIR = Path("data/nypl-reg/xml")
_REN_DIR = Path("data/nypl-ren/data")
_INDEX_PATH = Path("caches/cce.lmdb")
_RENEWAL_REVIEW_DB = Path("data/renewal_review.db")
_REVIEW_DB = Path("data/review.db")
_VAULT_PATH = Path("data/training/label_vault.jsonl")

_EXACT_DATE_BASELINE = 172_355
"""Renewal-centric exact-date joins from docs/findings/nypl_join_analysis_2026-06-27.md."""

_ADDITIONAL_ENTRY_BASELINE = 12_131
"""additionalEntry net-new over the exact-date baseline (additional_entry_join_yield)."""


def _log(message: str) -> None:
    """Append a timestamped milestone to the shared agent progress log."""
    with Path("/tmp/agent-progress.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{perf_counter():.0f}s post-rebuild: {message}\n")


def _pct(part: int, whole: int) -> str:
    """Return ``part/whole`` as a percentage string, guarding against zero."""
    if whole == 0:
        return "n/a"
    return f"{part / whole:.4%}"


def _exact_keys(regnum: str, iso_date: str) -> tuple[bytes, ...]:
    """Reconstruct the retired exact-date join keys (``normalize|isoformat``).

    Mirrors :func:`make_renewal_keys`'s multi-number fan-out but suffixes the
    full ISO date instead of the year, so the exact-date baseline the current
    codec no longer emits can be measured side-by-side with the year keys.
    """
    if not is_multi_regnum(regnum):
        return (f"{normalize_regnum(regnum)}|{iso_date}".encode(),)
    return tuple(f"{normalize_regnum(token)}|{iso_date}".encode() for token in regnum.split())


class _RenewalSide:
    """Renewal-side join keys and per-renewal identity, loaded once."""

    __slots__ = (
        "entry_and_id_to_ridx",
        "id_to_ridx",
        "joinable",
        "no_odat",
        "no_oreg",
        "odat",
        "oreg",
        "ren_key_set",
        "total",
    )

    def __init__(self) -> None:
        self.oreg: list[str | None] = []
        self.odat: list[date | None] = []
        self.ren_key_set: set[bytes] = set()
        self.entry_and_id_to_ridx: dict[tuple[str, str], int] = {}
        self.id_to_ridx: dict[str, int] = {}
        self.total = 0
        self.joinable = 0
        self.no_oreg = 0
        self.no_odat = 0


def _load_renewals(ren_dir: Path) -> _RenewalSide:
    """Stream every renewal once, recording production year keys and identity."""
    side = _RenewalSide()
    for record in iter_nypl_ren_directory(ren_dir):
        ridx = side.total
        side.total += 1
        side.oreg.append(record.oreg)
        side.odat.append(record.odat)
        side.entry_and_id_to_ridx[(record.entry_id, record.id)] = ridx
        side.id_to_ridx[record.id] = ridx
        if record.oreg is None:
            side.no_oreg += 1
        if record.odat is None:
            side.no_odat += 1
        if record.oreg is None or record.odat is None:
            continue
        side.joinable += 1
        keys = make_renewal_keys(record.oreg, record.odat.year)
        side.ren_key_set.update(keys)
        if side.total % 100_000 == 0:
            _log(f"renewals loaded {side.total:,}")
    return side


class _RegSide:
    """Complete registration keyspace plus reg-centric join + vault results."""

    __slots__ = (
        "add_year_keys",
        "regs_joined_complete",
        "regs_total",
        "regs_with_add",
        "top_exact_keys",
        "top_year_keys",
        "vault_joined",
    )

    def __init__(self) -> None:
        self.top_year_keys: set[bytes] = set()
        self.add_year_keys: set[bytes] = set()
        self.top_exact_keys: set[bytes] = set()
        self.regs_total = 0
        self.regs_with_add = 0
        self.regs_joined_complete = 0
        self.vault_joined: dict[str, bool] = {}


def _complete_keys(record: NyplRegRecord) -> tuple[bytes, ...]:
    """Return every join key a registration contributes (top-level + additional).

    Byte-identical to :func:`pd_matcher.index.builder._registration_join_keys`.
    """
    keys: list[bytes] = []
    if record.regnum is not None:
        keys.extend(make_renewal_keys(record.regnum, record.reg_year))
    for additional_regnum, additional_year in record.additional_join_keys:
        keys.extend(make_renewal_keys(additional_regnum, additional_year))
    return tuple(keys)


def _load_registrations(reg_dir: Path, rens: _RenewalSide, wanted_uuids: set[str]) -> _RegSide:
    """Stream every registration once, building the complete keyspace.

    Populates the three reg-side key sets (top-level year, additionalEntry
    year, top-level exact-date), the reg-centric join count against the
    renewal key set (reconciles with the index ``renewal_joins`` meta), and the
    per-vault-uuid joined status used for the #113 harvest.
    """
    side = _RegSide()
    ren_keys = rens.ren_key_set
    for record in iter_nypl_reg_directory(reg_dir):
        side.regs_total += 1
        complete = _complete_keys(record)
        if record.regnum is not None:
            side.top_year_keys.update(make_renewal_keys(record.regnum, record.reg_year))
            if record.reg_date is not None:
                side.top_exact_keys.update(_exact_keys(record.regnum, record.reg_date.isoformat()))
        if record.additional_join_keys:
            side.regs_with_add += 1
            for additional_regnum, additional_year in record.additional_join_keys:
                side.add_year_keys.update(make_renewal_keys(additional_regnum, additional_year))
        joined = any(key in ren_keys for key in complete)
        if joined:
            side.regs_joined_complete += 1
        if record.uuid in wanted_uuids:
            side.vault_joined[record.uuid] = joined
        if side.regs_total % 200_000 == 0:
            _log(f"registrations loaded {side.regs_total:,}")
    return side


class _RenewalJoinBreakdown:
    """Renewal-centric join tallies and the year-vs-additionalEntry gain split."""

    __slots__ = (
        "add_over_exact",
        "add_over_year",
        "gain_add_only",
        "gain_both",
        "gain_year_only",
        "joined_add",
        "joined_complete",
        "joined_exact",
        "joined_now_by_ridx",
        "joined_year_top",
    )

    def __init__(self) -> None:
        self.joined_exact = 0
        self.joined_year_top = 0
        self.joined_add = 0
        self.joined_complete = 0
        self.gain_year_only = 0
        self.gain_add_only = 0
        self.gain_both = 0
        self.add_over_exact = 0
        self.add_over_year = 0
        self.joined_now_by_ridx: list[bool] = []


def _classify_renewals(rens: _RenewalSide, regs: _RegSide) -> _RenewalJoinBreakdown:
    """Resolve each renewal against the complete keyspace and the exact baseline.

    ``joined_exact`` (top-level exact-date) is a subset of ``joined_year_top``
    (top-level year), which is a subset of ``joined_complete`` (year OR
    additionalEntry), so the gain over the exact baseline splits cleanly into
    year-recovered and additionalEntry-recovered renewals.
    """
    breakdown = _RenewalJoinBreakdown()
    top_year = regs.top_year_keys
    top_exact = regs.top_exact_keys
    add_year = regs.add_year_keys
    joined_flags: list[bool] = [False] * rens.total
    for ridx in range(rens.total):
        oreg = rens.oreg[ridx]
        odat = rens.odat[ridx]
        if oreg is None or odat is None:
            continue
        year_keys = make_renewal_keys(oreg, odat.year)
        exact_keys = _exact_keys(oreg, odat.isoformat())
        joined_exact = any(key in top_exact for key in exact_keys)
        joined_year_top = any(key in top_year for key in year_keys)
        joined_add = any(key in add_year for key in year_keys)
        joined_now = joined_year_top or joined_add
        joined_flags[ridx] = joined_now

        if joined_exact:
            breakdown.joined_exact += 1
        if joined_year_top:
            breakdown.joined_year_top += 1
        if joined_add:
            breakdown.joined_add += 1
        if joined_now:
            breakdown.joined_complete += 1

        if joined_add and not joined_exact:
            breakdown.add_over_exact += 1
        if joined_add and not joined_year_top:
            breakdown.add_over_year += 1

        if joined_now and not joined_exact:
            if joined_year_top and joined_add:
                breakdown.gain_both += 1
            elif joined_year_top:
                breakdown.gain_year_only += 1
            else:
                breakdown.gain_add_only += 1
    breakdown.joined_now_by_ridx = joined_flags
    return breakdown


class _Scenario4Recheck:
    """Bogus-scenario-4 tally over the renewal review DB."""

    __slots__ = ("now_joined", "still_orphan", "total", "unresolved")

    def __init__(self) -> None:
        self.total = 0
        self.unresolved = 0
        self.now_joined = 0
        self.still_orphan = 0


def _measure_scenario4(
    db_path: Path, rens: _RenewalSide, breakdown: _RenewalJoinBreakdown
) -> _Scenario4Recheck:
    """Count ``pairing_type='renewal'`` rows whose renewal now joins the keyspace.

    Each row is resolved to its renewal via ``(nypl_uuid, cce_renewal_id) ==
    (entry_id, renewal id)``, falling back to the renewal id alone. A row is a
    bogus scenario-4 when that renewal now joins the complete registration
    keyspace (a registration exists for it after all).
    """
    recheck = _Scenario4Recheck()
    connection = connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT nypl_uuid, cce_renewal_id FROM review_pair WHERE pairing_type = 'renewal'"
        ).fetchall()
    finally:
        connection.close()
    for entry_id, renewal_id in rows:
        recheck.total += 1
        ridx = rens.entry_and_id_to_ridx.get((entry_id, renewal_id))
        if ridx is None and renewal_id is not None:
            ridx = rens.id_to_ridx.get(renewal_id)
        if ridx is None:
            recheck.unresolved += 1
            continue
        if breakdown.joined_now_by_ridx[ridx]:
            recheck.now_joined += 1
        else:
            recheck.still_orphan += 1
    return recheck


class _VaultHarvest:
    """#113 harvest: verified reg-matches whose registration is now renewed."""

    __slots__ = (
        "newly_joined_vs_stamp",
        "now_joined",
        "unresolved",
        "verified_reg_matches",
        "was_renewed_stamped",
    )

    def __init__(self) -> None:
        self.verified_reg_matches = 0
        self.unresolved = 0
        self.now_joined = 0
        self.was_renewed_stamped = 0
        self.newly_joined_vs_stamp = 0


def _load_verified_reg_matches(vault_path: Path) -> list[VaultEntry]:
    """Return verified registration-pathway match entries from the vault (read-only).

    A verified registration match is ``verdict == "match"`` whose
    ``match_source`` is a registration pathway. Pre-schema-7 entries carry
    ``match_source is None``; the schema-7 migration backfills those to
    ``"registration"``, so ``None`` is treated as registration here. The
    ``"renewal"`` pathway is excluded — those are already MARC<->renewal pairs.
    """
    out: list[VaultEntry] = []
    for entry in iter_entries(vault_path):
        if entry.verdict != "match":
            continue
        if entry.match_source == "renewal":
            continue
        out.append(entry)
    return out


def _measure_vault_harvest(entries: list[VaultEntry], regs: _RegSide) -> _VaultHarvest:
    """Count verified reg-matches whose registration now joins a renewal."""
    harvest = _VaultHarvest()
    for entry in entries:
        harvest.verified_reg_matches += 1
        if entry.was_renewed:
            harvest.was_renewed_stamped += 1
        joined = regs.vault_joined.get(entry.nypl_uuid)
        if joined is None:
            harvest.unresolved += 1
            continue
        if joined:
            harvest.now_joined += 1
            if not entry.was_renewed:
                harvest.newly_joined_vs_stamp += 1
    return harvest


def _write_report(
    rens: _RenewalSide,
    regs: _RegSide,
    breakdown: _RenewalJoinBreakdown,
    scenario4: _Scenario4Recheck | None,
    scenario4_db: Path | None,
    harvest: _VaultHarvest,
    index_joins: int,
    index_regs: int,
    index_rens: int,
) -> None:
    """Print the numbers-first report answering questions 1-3."""
    write = stdout.write

    write("\n=== post-rebuild join-maximization verification (full corpus) ===\n")

    write("\n--- corpus + reconciliation ---\n")
    write(
        f"registrations parsed:            {regs.regs_total:,}  (index meta {index_regs:,}, "
        f"delta {regs.regs_total - index_regs:+,})\n"
    )
    write(
        f"renewals parsed:                 {rens.total:,}  (index meta {index_rens:,}, "
        f"delta {rens.total - index_rens:+,})\n"
    )
    write(f"joinable renewals (oreg+odat):   {rens.joinable:,}\n")
    write(f"  renewals with no oreg:         {rens.no_oreg:,}\n")
    write(f"  renewals with no odat:         {rens.no_odat:,}\n")
    write(f"regs with >=1 additionalEntry:   {regs.regs_with_add:,}\n")
    write(f"distinct top-level year keys:    {len(regs.top_year_keys):,}\n")
    write(f"distinct additionalEntry keys:   {len(regs.add_year_keys):,}\n")
    write(f"distinct top-level exact keys:   {len(regs.top_exact_keys):,}\n")
    write(
        f"reg-centric joins (complete):    {regs.regs_joined_complete:,}  "
        f"(index meta {index_joins:,}, delta {regs.regs_joined_complete - index_joins:+,})\n"
    )

    write("\n--- Q1: renewal-centric joins now ---\n")
    write(f"joined NOW (complete keyspace):  {breakdown.joined_complete:,}\n")
    write(f"joined via top-level year key:   {breakdown.joined_year_top:,}\n")
    write(f"joined via additionalEntry key:  {breakdown.joined_add:,}\n")
    write(
        f"exact-date baseline (recon):     {breakdown.joined_exact:,}  "
        f"(prior finding {_EXACT_DATE_BASELINE:,}, "
        f"delta {breakdown.joined_exact - _EXACT_DATE_BASELINE:+,})\n"
    )
    gain = breakdown.joined_complete - breakdown.joined_exact
    write(f"TOTAL gain over exact baseline:  {gain:+,}\n")
    write(f"  as % of exact baseline:        {_pct(gain, breakdown.joined_exact)}\n")

    write("\n--- Q1: gain breakdown (renewals joined NOW but NOT under exact-date) ---\n")
    write(f"  (a) recovered by YEAR only:               {breakdown.gain_year_only:,}\n")
    write(f"  (c) recovered by BOTH year + additional:  {breakdown.gain_both:,}\n")
    write(f"  (b) recovered by additionalEntry ONLY:    {breakdown.gain_add_only:,}\n")
    year_gain = breakdown.gain_year_only + breakdown.gain_both
    write(f"  year-attributable gain (a+c):             {year_gain:,}\n")
    write(f"  additionalEntry-attributable gain (b):    {breakdown.gain_add_only:,}\n")
    write(
        f"  sum (must equal total gain):              "
        f"{breakdown.gain_year_only + breakdown.gain_both + breakdown.gain_add_only:,}\n"
    )

    write("\n--- Q1: additionalEntry contribution confirmation ---\n")
    write(
        f"additionalEntry over EXACT baseline:  {breakdown.add_over_exact:,}  "
        f"(prior finding {_ADDITIONAL_ENTRY_BASELINE:,}, "
        f"delta {breakdown.add_over_exact - _ADDITIONAL_ENTRY_BASELINE:+,})\n"
    )
    write(
        f"additionalEntry marginal over YEAR:   {breakdown.add_over_year:,}  "
        f"(net-new renewals only additionalEntry recovers)\n"
    )
    landed = "YES" if breakdown.add_over_year > 0 else "NO -- RED FLAG"
    write(f"additionalEntry landed on renewal axis: {landed}\n")

    write("\n--- Q2: bogus scenario-4 recheck (renewal review DB) ---\n")
    if scenario4 is None or scenario4_db is None:
        write("no renewal review DB found; skipped\n")
    else:
        write(f"review DB:                       {scenario4_db}\n")
        write(f"pairing_type='renewal' rows:     {scenario4.total:,}\n")
        write(f"  resolved to a corpus renewal:  {scenario4.total - scenario4.unresolved:,}\n")
        write(f"  unresolved (no id match):      {scenario4.unresolved:,}\n")
        write(
            f"  NOW JOIN complete keyspace:    {scenario4.now_joined:,}  (were bogus scenario-4)\n"
        )
        write(f"  still genuine orphans:         {scenario4.still_orphan:,}\n")
        write(
            f"  bogus as % of resolved:        "
            f"{_pct(scenario4.now_joined, scenario4.total - scenario4.unresolved)}\n"
        )

    write("\n--- Q3: #113 harvest (free verified MARC<->renewal positives) ---\n")
    write(f"verified reg-pathway matches:    {harvest.verified_reg_matches:,}\n")
    write(
        f"  resolved to a corpus reg:      {harvest.verified_reg_matches - harvest.unresolved:,}\n"
    )
    write(f"  unresolved (uuid not in reg):  {harvest.unresolved:,}\n")
    write(f"  reg now joined (HARVESTABLE):  {harvest.now_joined:,}\n")
    write(
        f"  fraction of verified matches:  "
        f"{_pct(harvest.now_joined, harvest.verified_reg_matches)}\n"
    )
    write(f"  was_renewed already stamped:   {harvest.was_renewed_stamped:,}\n")
    write(f"  newly joined vs stamp:         {harvest.newly_joined_vs_stamp:,}\n")
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

    _log("start; reading vault (read-only)")
    vault_entries = _load_verified_reg_matches(_VAULT_PATH)
    wanted_uuids = {entry.nypl_uuid for entry in vault_entries}
    write(f"verified reg-pathway matches in vault: {len(vault_entries):,}\n")
    stdout.flush()

    _log("loading renewals")
    write("loading renewals ...\n")
    stdout.flush()
    rens = _load_renewals(ren_dir)
    write(f"  {rens.total:,} renewals in {perf_counter() - start:,.1f}s\n")
    _log(f"renewals done {rens.total:,}; parsing registrations (slow)")
    stdout.flush()

    reg_start = perf_counter()
    write("parsing registrations (slow pass) ...\n")
    stdout.flush()
    regs = _load_registrations(reg_dir, rens, wanted_uuids)
    write(f"  {regs.regs_total:,} registrations in {perf_counter() - reg_start:,.1f}s\n")
    _log(f"registrations done {regs.regs_total:,}; classifying renewals")
    stdout.flush()

    breakdown = _classify_renewals(rens, regs)
    _log("renewals classified; measuring scenario-4 + harvest")

    scenario4_db: Path | None = None
    if _RENEWAL_REVIEW_DB.exists():
        scenario4_db = _RENEWAL_REVIEW_DB
    elif _REVIEW_DB.exists():
        scenario4_db = _REVIEW_DB
    scenario4 = (
        _measure_scenario4(scenario4_db, rens, breakdown) if scenario4_db is not None else None
    )

    harvest = _measure_vault_harvest(vault_entries, regs)

    _write_report(
        rens,
        regs,
        breakdown,
        scenario4,
        scenario4_db,
        harvest,
        index_joins,
        index_regs,
        index_rens,
    )

    write(f"\ntotal runtime: {perf_counter() - start:,.1f}s\n")
    stdout.flush()
    _log(f"DONE total {perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
