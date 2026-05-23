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
from pd_matcher.config.loader import load_copyright_rules
from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.copyright.assessment import CopyrightAssessment
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

_LOGGER = getLogger(__name__)

_IDF_CACHE_NAME: str = "idf.msgpack"
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


def _evidence_payload(evidence: tuple[Evidence, ...]) -> dict[str, float]:
    """Return a ``scorer -> normalized score`` mapping for the 2b card."""
    return {ev.scorer: ev.normalized for ev in evidence if not ev.skipped}


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
    row using that vault entry's verdict / reasons / note / ``labeled_at`` so a
    rebuilt queue still reports the pair as labeled and ``next_unlabeled`` skips
    it. The reviewer can step back to re-label if needed.
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
    )

    def __init__(
        self,
        *,
        db_path: Path,
        budget: BudgetModel,
        seed: int,
        vault: dict[tuple[str, str], VaultEntry] | None = None,
    ) -> None:
        self._db_path = db_path
        self._budget = budget
        self._seed = seed
        self._vault: dict[tuple[str, str], VaultEntry] = vault or {}
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
        """Draw the below-0.7 reservoir, persist it, then commit and close."""
        db = self._require_db()
        if exc_type is None:
            self._finalize_below_sample(db)
            db.commit()
            self._log_fill()
            new_pairs = sum(self._kept.values()) - self._vault_applied
            _LOGGER.info(
                "vault contributed %d pre-applied labels; %d new pairs queued",
                self._vault_applied,
                new_pairs,
            )
        db.__exit__(exc_type, exc, tb)
        self._db = None

    def write(
        self,
        marc: MarcRecord,
        match: MatchResult | None,
        assessment: CopyrightAssessment,
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
            reasons=entry.reasons,
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
    budget, and the seed keeps every attribute picklable across ``spawn``.

    ``vault_path`` is read inside the writer process (so the parent never
    needs to ship the parsed vault across the process boundary) — the factory
    only carries the path.
    """

    db_path: Path
    budget: BudgetModel
    seed: int
    vault_path: Path

    def __call__(self, _output_path: Path) -> StratifyingResultWriter:
        """Construct the writer; ``_output_path`` is unused by design."""
        vault = current_entries(self.vault_path)
        return StratifyingResultWriter(
            db_path=self.db_path,
            budget=self.budget,
            seed=self.seed,
            vault=vault,
        )


class BuildSummary(Struct, frozen=True, forbid_unknown_fields=True):
    """Result of one :func:`build_queue` invocation."""

    records_sampled: int
    records_matched: int
    pairs_written: int
    stratum_counts: dict[str, int]


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
) -> list[MarcRecord]:
    """Reservoir-sample up to ``sample_per_lang`` MARC records from one dir."""

    def _records() -> Iterator[MarcRecord]:
        for shard in sorted(language_dir.glob("*.xml")):
            yield from iter_marc_records(shard)

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
    ruleset: CopyrightRuleSet,
    copyright_config: CopyrightAssessmentConfig,
    seed: int,
    workers: int,
    sample_per_lang: int,
    verbosity: int = 0,
) -> BuildSummary:
    """Build the stratified review queue and persist it to ``out_path``.

    Samples the pool per language, materializes the sample as prepared chunks,
    then drives :func:`pd_matcher.workers.run_match` with a
    :class:`StratifyingResultWriter`. The matcher's ``min_combined_score`` is
    forced to ``0.0`` so the writer sees every candidate's score and can band
    it; per-worker and aggregate throughput/ETA come from ``run_match``'s
    reporter, and the writer logs per-stratum fill.

    Args:
        pool: Root directory whose ``<lang>/*.xml`` shards form the pool.
        index_path: LMDB env produced by ``pd-matcher index build``.
        out_path: Destination SQLite review database.
        vault_path: JSONL label vault; current entries are pre-applied to the
            queue so previously-labeled pairs stay labeled across rebuilds.
        budget: Per-(language, band) caps.
        matching_config: Active config; the score floor is forced to ``0.0``.
        pairing_config: Active field-pairing config.
        ruleset: Loaded copyright ruleset for the matcher's assessment stage.
        copyright_config: Loaded copyright assessment config.
        seed: Seed for the reservoir samplers.
        workers: Number of matcher worker processes (``>= 1``).
        sample_per_lang: Reservoir size per language directory.
        verbosity: Forwarded to ``run_match`` (``-v`` worker heartbeats).

    Returns:
        A populated :class:`BuildSummary`.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1 (got {workers!r})")
    sampled: list[MarcRecord] = []
    for language, language_dir in _iter_language_dirs(pool):
        records = _sample_language(language_dir, sample_per_lang=sample_per_lang, seed=seed)
        _LOGGER.info("sampled %d records for language=%s", len(records), language)
        sampled.extend(records)

    prepared_dir = Path(mkdtemp(prefix=_PREPARED_PREFIX))
    try:
        manifest = _write_sample_chunks(sampled, prepared_dir)
        idf_cache_path = index_path.parent / _IDF_CACHE_NAME
        idf = load_or_build_idf(idf_cache_path, lambda: NyplIndexLookup(index_path))
        calibrator = _load_calibrator(index_path.parent)
        floored_config = replace(matching_config, min_combined_score=0.0)
        factory = StratifyingWriterFactory(
            db_path=out_path, budget=budget, seed=seed, vault_path=vault_path
        )
        _LOGGER.info("matching start: total=%d workers=%d", manifest.total_records, workers)
        report = run_match(
            prepared_dir=prepared_dir,
            expected_total=manifest.total_records,
            index_path=index_path,
            output_path=out_path,
            matching_config=floored_config,
            copyright_config=copyright_config,
            ruleset=ruleset,
            pairing_config=pairing_config,
            idf=idf,
            calibrator=calibrator,
            workers=workers,
            writer_factory=factory,
            verbosity=verbosity,
        )
    finally:
        rmtree(prepared_dir, ignore_errors=True)

    counts = _read_stratum_counts(out_path)
    labeled = {f"{language}/{band}": n for (language, band), n in counts.items()}
    for label, n in sorted(labeled.items()):
        _LOGGER.info("stratum %s filled=%d", label, n)
    pairs_written = sum(labeled.values())
    _LOGGER.info(
        "build complete: sampled=%d matched=%d written=%d",
        len(sampled),
        report.records_processed,
        pairs_written,
    )
    return BuildSummary(
        records_sampled=len(sampled),
        records_matched=report.records_processed,
        pairs_written=pairs_written,
        stratum_counts=labeled,
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


def load_default_ruleset() -> CopyrightRuleSet:
    """Load the shipped copyright ruleset for the matcher's assessment stage."""
    from importlib.resources import as_file
    from importlib.resources import files

    resource = files("pd_matcher.config.defaults") / "copyright_rules.yaml"
    with as_file(resource) as path:
        return load_copyright_rules(Path(path))


__all__ = [
    "BuildSummary",
    "StratifyingResultWriter",
    "StratifyingWriterFactory",
    "build_queue",
    "load_default_ruleset",
]
