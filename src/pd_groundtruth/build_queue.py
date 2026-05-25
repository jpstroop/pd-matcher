"""Stratified review-queue builder orchestration.

Runs the main app's matcher (:mod:`pd_matcher`) over a stratified sample of
the full-MARC candidate pool and writes a self-contained SQLite review DB of
proposed ``(MARC, CCE-candidate)`` pairs for Phase 2b hand-labeling.

There is exactly one matching path in this codebase: the spawn pool owned by
:func:`pd_matcher.workers.run_match`. This module reuses it. It samples the
pool per language, materializes the sample as prepared pickled chunks (so the
matcher's chunk-replay source can drive its reporter with a known total),
then hands ``run_match`` a custom :class:`StratifyingResultWriter` via a
picklable :class:`StratifyingWriterFactory`. The matcher runs with
``min_combined_score = 0.0`` so the writer observes every candidate's score
and can band it; the writer accepts banded pairs greedily until each
``(language, band)`` cap fills and buffers below-0.7 candidates for a seeded
reservoir draw at close.

The writer is the only party that touches the SQLite file. ``run_match``
spawns a single writer process and invokes the factory inside it, so there is
never concurrent access to the database. The factory therefore carries only
picklable values (paths, the budget, the seed) across the process boundary.
"""

from collections.abc import Iterator
from datetime import UTC
from datetime import date
from datetime import datetime
from logging import getLogger
from pathlib import Path
from pickle import HIGHEST_PROTOCOL
from pickle import dump
from shutil import rmtree
from tempfile import mkdtemp
from types import TracebackType
from typing import Self

from msgspec import Struct
from msgspec.json import encode as json_encode
from msgspec.structs import replace

from pd_groundtruth.build_queue_vault import resolve_vault_for_build
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.progress import render_kept_suffix
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import BAND_BELOW
from pd_groundtruth.sampling import SOURCE_BANDED
from pd_groundtruth.sampling import SOURCE_BELOW_SAMPLE
from pd_groundtruth.sampling import BudgetModel
from pd_groundtruth.sampling import band_of
from pd_groundtruth.sampling import reservoir_sample
from pd_groundtruth.vault_pair_resolver import IDF_CACHE_NAME as _SHARED_IDF_CACHE_NAME
from pd_groundtruth.vault_pair_resolver import ResolvedVaultPair
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.calibrator import load_calibrator
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.prepare import PreparedManifest
from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.parsers.marc import iter_marc_records
from pd_matcher.workers import run_match

_LOGGER = getLogger(__name__)

_IDF_CACHE_NAME: str = _SHARED_IDF_CACHE_NAME
_CALIBRATOR_NAME: str = "calibrator.msgpack"
_PREPARED_PREFIX: str = "pd-groundtruth-prepared-"
_DEFAULT_LANGUAGE: str = "eng"
_FILL_LOG_INTERVAL: int = 250


def _decade_of(year: int | None) -> int | None:
    """Return the decade bucket for ``year`` (e.g. 1953 -> 1950)."""
    if year is None:
        return None
    return (year // 10) * 10


def _join(values: tuple[str, ...]) -> str | None:
    """Join a tuple of strings with ``" | "`` or return ``None`` when empty."""
    return " | ".join(values) if values else None


def _join_places(values: tuple[str, ...]) -> str | None:
    """Join a tuple of publication-place strings with ``"; "`` or return ``None``."""
    return "; ".join(values) if values else None


def _join_notes(values: tuple[str, ...]) -> str | None:
    """Join a tuple of CCE notes with newlines or return ``None`` when empty."""
    return "\n".join(values) if values else None


def _join_prev_regnums(values: tuple[str, ...]) -> str | None:
    """Join a tuple of CCE prev-regnums with ``"; "`` or return ``None``."""
    return "; ".join(values) if values else None


def _iso_or_none(value: date | None) -> str | None:
    """Return ``value.isoformat()`` or ``None`` when the date is absent."""
    return value.isoformat() if value is not None else None


def _evidence_payload(evidence: tuple[Evidence, ...]) -> dict[str, float]:
    """Return a ``scorer -> normalized score`` mapping for the 2b card."""
    return {ev.scorer: ev.normalized for ev in evidence if not ev.skipped}


_SOURCE_SEPARATOR: str = " ↔ "


def _evidence_sources_payload(
    evidence: tuple[Evidence, ...],
    sources: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    """Return a ``scorer -> "marc_field ↔ cce_field"`` mapping for the 2b card.

    Sources from non-group scorers (lccn, isbn, year, edition) carry the
    sentinel ``("", "")`` pair from the pipeline and are dropped here; the
    card already implies their field pairing from the scorer name. Skipped
    Evidence is dropped to mirror :func:`_evidence_payload`. Length mismatch
    between ``evidence`` and ``sources`` (e.g. an Evidence tuple from a
    pre-#50 cache) collapses to an empty payload so the persisted column is
    still valid JSON.
    """
    if len(sources) != len(evidence):
        return {}
    payload: dict[str, str] = {}
    for ev, source in zip(evidence, sources, strict=True):
        if ev.skipped:
            continue
        marc_field, cce_field = source
        if not marc_field or not cce_field:
            continue
        payload[ev.scorer] = f"{marc_field}{_SOURCE_SEPARATOR}{cce_field}"
    return payload


def _language_of(marc: MarcRecord) -> str:
    """Return the MARC language code, falling back to :data:`_DEFAULT_LANGUAGE`."""
    return marc.language_code or _DEFAULT_LANGUAGE


class _BufferedCandidate(Struct, frozen=True, forbid_unknown_fields=True):
    """A below-0.7 candidate held in memory until the close-time reservoir draw."""

    language: str
    score: float
    pair: PairInsert


def _build_pair_insert(
    marc: MarcRecord,
    matched_nypl: IndexedNyplRegRecord,
    evidence: tuple[Evidence, ...],
    *,
    language: str,
    score: float,
    band: str,
    source: str,
    evidence_sources: tuple[tuple[str, str], ...] = (),
) -> PairInsert:
    """Assemble a :class:`PairInsert` snapshot from one matched record."""
    return PairInsert(
        language=language,
        decade=_decade_of(marc.publication_year),
        score=score,
        band=band,
        source=source,
        marc_control_id=marc.control_id,
        marc_json=json_encode(marc).decode("utf-8"),
        marc_title=marc.title,
        marc_author=marc.main_author or marc.statement_of_responsibility,
        marc_publisher=marc.publisher,
        marc_year=marc.publication_year,
        nypl_uuid=matched_nypl.uuid,
        cce_title=matched_nypl.title,
        cce_author=matched_nypl.author_name,
        cce_publishers=_join(matched_nypl.publisher_names),
        cce_claimants=_join(matched_nypl.claimants),
        cce_reg_year=matched_nypl.reg_year,
        cce_was_renewed=matched_nypl.was_renewed,
        cce_regnum=matched_nypl.regnum,
        evidence_json=json_encode(_evidence_payload(evidence)).decode("utf-8"),
        cce_edition=matched_nypl.edition,
        cce_publication_places=_join_places(matched_nypl.publication_places),
        cce_author_place=matched_nypl.author_place,
        cce_author_is_claimant=matched_nypl.author_is_claimant,
        cce_copies=matched_nypl.copies,
        cce_aff_date=_iso_or_none(matched_nypl.aff_date),
        cce_desc=matched_nypl.desc,
        cce_notes=_join_notes(matched_nypl.notes),
        cce_new_matter_claimed=matched_nypl.new_matter_claimed,
        cce_copy_date=_iso_or_none(matched_nypl.copy_date),
        cce_notice_date=_iso_or_none(matched_nypl.notice_date),
        cce_lccn=matched_nypl.lccn,
        cce_prev_regnums=_join_prev_regnums(matched_nypl.prev_regnums),
        cce_renewal_id=matched_nypl.renewal_id,
        cce_renewal_oreg=matched_nypl.renewal_oreg,
        cce_renewal_rdat=_iso_or_none(matched_nypl.renewal_rdat),
        cce_renewal_author=matched_nypl.renewal_author,
        cce_renewal_title=matched_nypl.renewal_title,
        cce_renewal_claimants=matched_nypl.renewal_claimants,
        cce_renewal_new_matter=matched_nypl.renewal_new_matter,
        evidence_sources_json=json_encode(
            _evidence_sources_payload(evidence, evidence_sources)
        ).decode("utf-8"),
    )


class StratifyingResultWriter:
    """A :class:`pd_matcher.output.csv_writer.ResultWriter` that stratifies.

    Constructed inside the matcher's writer process from a
    :class:`StratifyingWriterFactory`. Every matched record arrives once via
    :meth:`write`; banded acceptances (``>=0.7``) are inserted immediately up
    to their ``(language, band)`` cap, and below-0.7 candidates are buffered
    so :meth:`__exit__` can draw a seeded reservoir per language for the
    :data:`pd_groundtruth.sampling.SOURCE_BELOW_SAMPLE` bucket. Order does not
    matter, so this works under ``imap``-style unordered delivery.

    When a vault entry exists for an inserted pair's
    ``(marc_control_id, nypl_uuid)`` key, the writer also inserts a ``label``
    row using that vault entry's verdict / note / ``labeled_at`` so a
    rebuilt queue still reports the pair as labeled and ``next_unlabeled`` skips
    it. The reviewer can step back to re-label if needed.

    Pre-resolved vault pairs (already scored against the matcher's per-pair
    routine by :func:`pd_groundtruth.vault_pair_resolver.resolve_vault_pairs`
    in the parent) are injected unconditionally on close, bypassing the
    per-stratum caps so every persistable vault verdict makes it back into
    the rebuilt queue regardless of sample size.
    """

    __slots__ = (
        "_below_buffer",
        "_budget",
        "_db",
        "_db_path",
        "_kept",
        "_seed",
        "_seen",
        "_vault",
        "_vault_applied",
        "_vault_pairs",
    )

    def __init__(
        self,
        *,
        db_path: Path,
        budget: BudgetModel,
        seed: int,
        vault: dict[tuple[str, str], VaultEntry] | None = None,
        vault_pairs: tuple[ResolvedVaultPair, ...] = (),
    ) -> None:
        self._db_path = db_path
        self._budget = budget
        self._seed = seed
        self._vault: dict[tuple[str, str], VaultEntry] = vault or {}
        self._vault_pairs: tuple[ResolvedVaultPair, ...] = vault_pairs
        self._db: ReviewDb | None = None
        self._kept: dict[tuple[str, str], int] = {}
        self._below_buffer: dict[str, list[_BufferedCandidate]] = {}
        self._seen: int = 0
        self._vault_applied: int = 0

    def __enter__(self) -> Self:
        """Open (creating schema) the review database."""
        self._db = ReviewDb.connect(self._db_path)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Draw the below-0.7 reservoir, inject vault pairs, commit and close."""
        db = self._require_db()
        if exc_type is None:
            self._finalize_below_sample(db)
            self._inject_vault_pairs(db)
            db.commit()
            self._log_fill()
            new_pairs = sum(self._kept.values()) - self._vault_applied
            _LOGGER.info(
                "vault pre-applied: %d pairs (of %d resolved); %d non-vault pairs queued",
                self._vault_applied,
                len(self._vault_pairs),
                new_pairs,
            )
        db.__exit__(exc_type, exc, tb)
        self._db = None

    def write(
        self,
        marc: MarcRecord,
        match: MatchResult | None,
        matched_nypl: IndexedNyplRegRecord | None = None,
    ) -> None:
        """Band one matched record and persist or buffer it accordingly."""
        db = self._require_db()
        if match is None or match.best is None or matched_nypl is None:
            return
        self._seen += 1
        language = _language_of(marc)
        score = match.best.combined.calibrated
        band = band_of(score)
        if band == BAND_BELOW:
            pair = _build_pair_insert(
                marc,
                matched_nypl,
                match.best.evidence,
                language=language,
                score=score,
                band=BAND_BELOW,
                source=SOURCE_BELOW_SAMPLE,
                evidence_sources=match.best.evidence_sources,
            )
            self._below_buffer.setdefault(language, []).append(
                _BufferedCandidate(language=language, score=score, pair=pair)
            )
            return
        key = (language, band)
        if self._kept.get(key, 0) >= self._budget.cap_for(language, band):
            return
        pair = _build_pair_insert(
            marc,
            matched_nypl,
            match.best.evidence,
            language=language,
            score=score,
            band=band,
            source=SOURCE_BANDED,
            evidence_sources=match.best.evidence_sources,
        )
        self._insert_with_vault(db, pair)
        self._kept[key] = self._kept.get(key, 0) + 1
        if self._seen % _FILL_LOG_INTERVAL == 0:
            self._log_fill()

    def _finalize_below_sample(self, db: ReviewDb) -> None:
        """Draw and persist the per-language below-0.7 reservoir."""
        for language, candidates in self._below_buffer.items():
            cap = self._budget.cap_for(language, BAND_BELOW)
            language_seed = self._seed ^ hash(language)
            for candidate in reservoir_sample(candidates, cap, language_seed):
                self._insert_with_vault(db, candidate.pair)
                key = (language, BAND_BELOW)
                self._kept[key] = self._kept.get(key, 0) + 1

    def _inject_vault_pairs(self, db: ReviewDb) -> None:
        """Persist every pre-resolved vault pair, bypassing per-stratum caps.

        Vault pairs are inserted *after* the regular stratifying flow has
        finalized, so cap counts reflect only matcher-proposed acceptances
        and the vault contribution stays a pure bonus. Each insertion also
        pre-applies the vault entry's verdict via
        :meth:`ReviewDb.insert_existing_label` so a rebuilt queue carries
        the original ``labeled_at`` forward unchanged.
        """
        for resolved in self._vault_pairs:
            pair_id = db.insert_pair(resolved.pair)
            db.insert_existing_label(
                pair_id=pair_id,
                verdict=resolved.entry.verdict,
                labeled_at=resolved.entry.labeled_at,
                note=resolved.entry.note,
            )
            self._vault_applied += 1
            key = (resolved.pair.language, resolved.pair.band)
            self._kept[key] = self._kept.get(key, 0) + 1

    def _insert_with_vault(self, db: ReviewDb, pair: PairInsert) -> None:
        """Insert one pair; if the vault has a verdict for it, also pre-apply it."""
        pair_id = db.insert_pair(pair)
        entry = self._vault.get((pair.marc_control_id, pair.nypl_uuid))
        if entry is None:
            return
        db.insert_existing_label(
            pair_id=pair_id,
            verdict=entry.verdict,
            labeled_at=entry.labeled_at,
            note=entry.note,
        )
        self._vault_applied += 1

    def _log_fill(self) -> None:
        """Emit the kept-per-stratum fill readout."""
        _LOGGER.info("queue.fill %s", render_kept_suffix(self._budget, self._kept))

    def _require_db(self) -> ReviewDb:
        """Return the open database or raise if used outside its context."""
        if self._db is None:
            raise RuntimeError("StratifyingResultWriter not entered; use as a context manager")
        return self._db


class StratifyingWriterFactory(Struct, frozen=True, forbid_unknown_fields=True):
    """Picklable factory that builds a :class:`StratifyingResultWriter`.

    ``run_match`` invokes the factory inside the spawned writer process with
    the run's ``output_path``; that path is ignored here because the review
    database location travels with the factory. Carrying only paths, the
    budget, the seed, and the already-resolved vault pairs keeps every
    attribute picklable across ``spawn``.

    ``vault_path`` is re-read inside the writer process so the matcher-route
    safety net still pre-applies labels for any pair the matcher happens to
    propose for a vault key (rare once vault MARCs are excluded from the
    sample, but kept as a belt-and-braces). ``vault_pairs`` carries the
    parent's pre-scored, ready-to-insert vault pairs across the boundary.
    """

    db_path: Path
    budget: BudgetModel
    seed: int
    vault_path: Path
    vault_pairs: tuple[ResolvedVaultPair, ...] = ()

    def __call__(self, _output_path: Path) -> StratifyingResultWriter:
        """Construct the writer; ``_output_path`` is unused by design."""
        vault = current_entries(self.vault_path)
        return StratifyingResultWriter(
            db_path=self.db_path,
            budget=self.budget,
            seed=self.seed,
            vault=vault,
            vault_pairs=self.vault_pairs,
        )


class BuildSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """Result of one :func:`build_queue` invocation.

    ``vault_resolved`` is the count of vault entries materialized into a
    persistable pair this run; ``vault_missing_in_pool`` and
    ``vault_missing_in_index`` are the diagnostic counts for vault entries
    that could not be carried forward this run because the underlying MARC or
    CCE record is no longer available (the vault file is never modified).
    """

    records_sampled: int
    records_matched: int
    pairs_written: int
    stratum_counts: dict[str, int]
    vault_resolved: int = 0
    vault_missing_in_pool: int = 0
    vault_missing_in_index: int = 0


def _iter_language_dirs(pool: Path) -> Iterator[tuple[str, Path]]:
    """Yield ``(language, dir)`` pairs for each language subdir under ``pool``."""
    for child in sorted(pool.iterdir()):
        if child.is_dir():
            yield child.name, child


def _sample_language(
    language_dir: Path,
    *,
    sample_per_lang: int,
    seed: int,
    exclude: frozenset[str] = frozenset(),
) -> list[MarcRecord]:
    """Reservoir-sample up to ``sample_per_lang`` records, skipping ``exclude``.

    Records whose ``control_id`` is in ``exclude`` are dropped before reservoir
    selection so they neither count against the sample budget nor reach the
    matcher. The intended use is to bypass vault-claimed MARCs (whose
    matcher-route output would be redundant with the pre-resolved vault
    pair already queued for insertion).
    """

    def _records() -> Iterator[MarcRecord]:
        for shard in sorted(language_dir.glob("*.xml")):
            for record in iter_marc_records(shard):
                if record.control_id in exclude:
                    continue
                yield record

    return reservoir_sample(_records(), sample_per_lang, seed)


def _write_sample_chunks(records: list[MarcRecord], out_dir: Path) -> PreparedManifest:
    """Pickle the in-memory ``records`` into ``out_dir`` as a single prepared chunk.

    The sample already lives in memory, so it is pickled directly with the
    same chunk codec ``run_match`` replays rather than round-tripped back
    through MARCXML. The returned manifest carries the record total so the
    matcher's reporter can show percent-complete and an ETA.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_name = "chunk_00000.pkl"
    with (out_dir / chunk_name).open("wb") as handle:
        dump(tuple(records), handle, protocol=HIGHEST_PROTOCOL)
    manifest = PreparedManifest(
        version=1,
        total_records=len(records),
        chunk_files=(chunk_name,),
        chunk_size=max(1, len(records)),
        source_hash="in-memory-sample",
        created_at=datetime.now(UTC).isoformat(),
    )
    (out_dir / "manifest.json").write_bytes(json_encode(manifest))
    return manifest


def build_queue(
    *,
    pool: Path,
    index_path: Path,
    out_path: Path,
    vault_path: Path,
    budget: BudgetModel,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    seed: int,
    workers: int,
    sample_per_lang: int,
    verbosity: int = 0,
    log_file: Path | None = None,
) -> BuildSummary:
    """Build the stratified review queue and persist it to ``out_path``.

    Resolves every current vault entry up front into a pre-scored pair so the
    rebuilt queue *always* carries the existing verdict forward (the matcher's
    candidate retrieval doesn't necessarily surface a previously-labeled pair
    on a fresh pass — see jpstroop/pd-matcher#33). Vault-claimed MARC records
    are then excluded from the per-language reservoir so the matcher does not
    re-propose them, and the writer inserts the resolved vault pairs at close
    *outside* the per-stratum caps so vault carryover is unconditional.

    Samples the (vault-excluded) pool per language, materializes the sample as
    prepared chunks, and drives :func:`pd_matcher.workers.run_match` with a
    :class:`StratifyingResultWriter`. The matcher's ``min_combined_score`` is
    forced to ``0.0`` so the writer sees every candidate's score and can band
    it; per-worker and aggregate throughput/ETA come from ``run_match``'s
    reporter, and the writer logs per-stratum fill.

    Args:
        pool: Root directory whose ``<lang>/*.xml`` shards form the pool.
        index_path: LMDB env produced by ``pd-matcher index build``.
        out_path: Destination SQLite review database.
        vault_path: JSONL label vault; current entries are resolved against
            the pool/index and pre-applied to the queue so previously-labeled
            pairs stay labeled across rebuilds, regardless of sample size.
        budget: Per-(language, band) caps.
        matching_config: Active config; the score floor is forced to ``0.0``.
        pairing_config: Active field-pairing config.
        seed: Seed for the reservoir samplers.
        workers: Number of matcher worker processes (``>= 1``).
        sample_per_lang: Reservoir size per language directory.
        verbosity: Forwarded to ``run_match`` (``-v`` worker heartbeats).
        log_file: Optional log file path forwarded to ``run_match`` so spawn
            workers append their log lines to the same file as the parent.

    Returns:
        A populated :class:`BuildSummary`.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1 (got {workers!r})")

    idf_cache_path = index_path.parent / _IDF_CACHE_NAME
    idf = load_or_build_idf(idf_cache_path, lambda: NyplIndexLookup(index_path))
    calibrator = _load_calibrator(index_path.parent)

    resolved_vault_pairs, vault_summary = resolve_vault_for_build(
        vault_path=vault_path,
        pool=pool,
        index_path=index_path,
        matching_config=matching_config,
        pairing_config=pairing_config,
        idf=idf,
        calibrator=calibrator,
    )
    exclude_ids = frozenset(resolved.pair.marc_control_id for resolved in resolved_vault_pairs)

    sampled: list[MarcRecord] = []
    for language, language_dir in _iter_language_dirs(pool):
        records = _sample_language(
            language_dir,
            sample_per_lang=sample_per_lang,
            seed=seed,
            exclude=exclude_ids,
        )
        _LOGGER.info("sampled %d records for language=%s", len(records), language)
        sampled.extend(records)

    prepared_dir = Path(mkdtemp(prefix=_PREPARED_PREFIX))
    try:
        manifest = _write_sample_chunks(sampled, prepared_dir)
        floored_config = replace(matching_config, min_combined_score=0.0)
        factory = StratifyingWriterFactory(
            db_path=out_path,
            budget=budget,
            seed=seed,
            vault_path=vault_path,
            vault_pairs=tuple(resolved_vault_pairs),
        )
        _LOGGER.info("matching start: total=%d workers=%d", manifest.total_records, workers)
        report = run_match(
            prepared_dir=prepared_dir,
            expected_total=manifest.total_records,
            index_path=index_path,
            output_path=out_path,
            matching_config=floored_config,
            pairing_config=pairing_config,
            idf=idf,
            calibrator=calibrator,
            workers=workers,
            writer_factory=factory,
            verbosity=verbosity,
            log_file=log_file,
        )
    finally:
        rmtree(prepared_dir, ignore_errors=True)

    counts = _read_stratum_counts(out_path)
    labeled = {f"{language}/{band}": n for (language, band), n in counts.items()}
    for label, n in sorted(labeled.items()):
        _LOGGER.info("stratum %s filled=%d", label, n)
    pairs_written = sum(labeled.values())
    _LOGGER.info(
        "build complete: sampled=%d matched=%d written=%d vault_resolved=%d",
        len(sampled),
        report.records_processed,
        pairs_written,
        vault_summary.resolved,
    )
    return BuildSummary(
        records_sampled=len(sampled),
        records_matched=report.records_processed,
        pairs_written=pairs_written,
        stratum_counts=labeled,
        vault_resolved=vault_summary.resolved,
        vault_missing_in_pool=vault_summary.missing_in_pool,
        vault_missing_in_index=vault_summary.missing_in_index,
    )


def _load_calibrator(parent: Path) -> PlattCalibrator | None:
    """Load a Platt calibrator from ``<parent>/calibrator.msgpack`` if present."""
    candidate = parent / _CALIBRATOR_NAME
    if not candidate.exists():
        return None
    return load_calibrator(candidate)


def _read_stratum_counts(out_path: Path) -> dict[tuple[str, str], int]:
    """Return persisted ``(language, band)`` counts from the review database."""
    if not out_path.exists():
        return {}
    with ReviewDb.connect(out_path) as db:
        return db.stratum_counts()


__all__ = [
    "BuildSummary",
    "StratifyingResultWriter",
    "StratifyingWriterFactory",
    "build_queue",
]
