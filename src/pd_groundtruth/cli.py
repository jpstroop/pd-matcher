"""Typer entry point for the ground-truth acquisition tool."""

from datetime import UTC
from datetime import datetime
from logging import INFO
from logging import FileHandler
from logging import Formatter
from logging import StreamHandler
from logging import getLogger
from pathlib import Path
from typing import Annotated

from msgspec.json import decode as json_decode
from typer import BadParameter
from typer import Exit
from typer import Option
from typer import Typer
from typer import echo

from pd_groundtruth.acquire import acquire
from pd_groundtruth.acquire import default_min_year
from pd_groundtruth.build_queue import build_queue
from pd_groundtruth.dump_vault_marcs import dump_vault_marcs
from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.label_vault import extract_marc_identifiers
from pd_groundtruth.label_vault import upsert_entry
from pd_groundtruth.manifest import DEFAULT_MANIFEST_URL
from pd_groundtruth.publish_linkage import publish_linkage
from pd_groundtruth.review.server import serve
from pd_groundtruth.review_db import VERDICT_MATCH
from pd_groundtruth.review_db import VERDICT_NO_MATCH
from pd_groundtruth.review_db import VERDICT_UNSURE
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import default_budget
from pd_groundtruth.sampling import scale_budget
from pd_groundtruth.vault_into_queue import vault_into_queue
from pd_groundtruth.vault_migration import migrate_vault_v3
from pd_groundtruth.vault_migration import migrate_vault_v4
from pd_groundtruth.vault_migration import migrate_vault_v5
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.models import MarcRecord

app = Typer(add_completion=False, help="Acquire Princeton MARC ground-truth candidates.")

_DEFAULT_PER_DECADE_CAP = 20000
_DEFAULT_SEED = 42
_DEFAULT_WORKERS = 8
_DEFAULT_SAMPLE_PER_LANG = 1500
_DEFAULT_REVIEW_HOST = "127.0.0.1"
_DEFAULT_REVIEW_PORT = 8000
_DEFAULT_POOL_PATH = Path("data/candidates")
_DEFAULT_INDEX_PATH = Path("caches/cce.lmdb")
_DEFAULT_REVIEW_DB_PATH = Path("data/review.db")
_DEFAULT_VAULT_PATH = Path("data/label_vault.jsonl")
_DEFAULT_VAULT_MARCS_PATH = Path("data/published/marc.xml")
_DEFAULT_PUBLISHED_TRAINING_PATH = Path("data/published/training.jsonl")
_DEFAULT_PUBLISHED_MATCHES_PATH = Path("data/published/matches.jsonl")
_LABELER = "jpstroop"
_LOG_DIR_NAME = "logs"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_REQUEUE_VALID_VERDICTS: frozenset[str] = frozenset(
    {VERDICT_MATCH, VERDICT_NO_MATCH, VERDICT_UNSURE}
)


def _utc_timestamp() -> str:
    """Return a filename-safe UTC timestamp (e.g. ``20260523-004530``)."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _configure_logging(command: str, log_file: Path | None) -> Path:
    """Install stdout + file log handlers and return the resolved file path.

    Resolves the path to ``log_file`` when supplied or
    ``logs/{command}_{utc-timestamp}.log`` otherwise, creates the parent
    directory, wires up one stream handler and one file handler at INFO
    with a uniform format string, and echoes the path to stderr so users
    see where the run is being persisted.
    """
    path = (
        log_file
        if log_file is not None
        else Path(_LOG_DIR_NAME) / (f"{command}_{_utc_timestamp()}.log")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = Formatter(_LOG_FORMAT)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root = getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.setLevel(INFO)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    echo(f"logging to file: {path}", err=True)
    return path


@app.callback()
def main() -> None:
    """Princeton MARC ground-truth acquisition CLI."""


@app.command(name="acquire")
def acquire_command(
    out_dir: Annotated[
        Path, Option("--out-dir", help="Root directory; shards written to <out-dir>/<lang>/.")
    ],
    manifest_url: Annotated[
        str, Option("--manifest-url", help="Dump manifest JSON URL.")
    ] = DEFAULT_MANIFEST_URL,
    per_decade_cap: Annotated[
        int,
        Option(
            "--per-decade-cap",
            help="Maximum records to keep per (language, decade) bucket.",
        ),
    ] = _DEFAULT_PER_DECADE_CAP,
    min_year: Annotated[
        int | None,
        Option(
            "--min-year",
            help="Lower bound for publication year (the moving wall, today.year - 95).",
        ),
    ] = None,
    max_dumps: Annotated[
        int | None, Option("--max-dumps", help="Cap the number of dumps processed.")
    ] = None,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Stream dumps and write eligible records as per-language MARCXML shards."""
    _configure_logging("acquire", log_file)
    resolved_min_year = default_min_year() if min_year is None else min_year
    report = acquire(
        out_dir=out_dir,
        per_decade_cap=per_decade_cap,
        min_year=resolved_min_year,
        manifest_url=manifest_url,
        max_dumps=max_dumps,
    )
    kept = " ".join(f"{language}={count}" for language, count in report.kept_by_language.items())
    echo(
        f"dumps_processed={report.dumps_processed} "
        f"records_scanned={report.records_scanned} "
        f"kept_by_language=[{kept}] "
        f"shards_written={report.shards_written} "
        f"stopped_reason={report.stopped_reason}"
    )


def _existing_pair_count(path: Path) -> int:
    """Return the number of ``review_pair`` rows in ``path``, or 0 if absent/empty.

    Treats any non-existent file, empty file, or file lacking the ``review_pair``
    table as having zero pairs; this keeps fresh-file and schema-only-file runs
    on the no-flag happy path.
    """
    if not path.exists() or path.stat().st_size == 0:
        return 0
    from sqlite3 import OperationalError
    from sqlite3 import connect as sqlite_connect

    connection = sqlite_connect(path)
    try:
        try:
            row = connection.execute("SELECT COUNT(*) FROM review_pair").fetchone()
        except OperationalError:
            return 0
        return int(row[0])
    finally:
        connection.close()


@app.command(name="build-queue")
def build_queue_command(
    pool: Annotated[
        Path,
        Option("--pool", help="Root dir whose <lang>/*.xml shards form the candidate pool."),
    ] = _DEFAULT_POOL_PATH,
    index: Annotated[
        Path, Option("--index", help="LMDB env produced by `pd-matcher index build`.")
    ] = _DEFAULT_INDEX_PATH,
    out: Annotated[
        Path, Option("--out", help="Destination SQLite review database.")
    ] = _DEFAULT_REVIEW_DB_PATH,
    vault: Annotated[
        Path,
        Option(
            "--vault",
            help="JSONL label vault; existing verdicts are pre-applied to the queue.",
        ),
    ] = _DEFAULT_VAULT_PATH,
    budget: Annotated[
        int | None,
        Option("--budget", help="Target total pairs; scales the default caps proportionally."),
    ] = None,
    seed: Annotated[int, Option("--seed", help="Seed for the reservoir samplers.")] = _DEFAULT_SEED,
    workers: Annotated[
        int, Option("--workers", help="Number of spawn-pool worker processes.")
    ] = _DEFAULT_WORKERS,
    sample_per_lang: Annotated[
        int,
        Option(
            "--sample-per-lang",
            help="Reservoir size per language directory (default fills the default budget).",
        ),
    ] = _DEFAULT_SAMPLE_PER_LANG,
    verbose: Annotated[
        int,
        Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase matcher logging: -v per-worker heartbeats, -vv per-record hits.",
        ),
    ] = 0,
    rebuild: Annotated[
        bool,
        Option(
            "--rebuild",
            help="Delete the target --out database before writing (destructive).",
        ),
    ] = False,
    append: Annotated[
        bool,
        Option(
            "--append",
            help="Append to a non-empty --out database (today's silent behavior, now opt-in).",
        ),
    ] = False,
    requeue: Annotated[
        list[str] | None,
        Option(
            "--requeue",
            help=(
                "Skip pre-applying vault entries with this verdict so the pair re-enters "
                "the queue. Repeatable. Valid: match, no_match, unsure."
            ),
        ),
    ] = None,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Match a stratified pool sample and write a SQLite review queue."""
    if rebuild and append:
        echo("--rebuild and --append are mutually exclusive.", err=True)
        raise Exit(code=2)
    requeue_values = frozenset(requeue or ())
    invalid = sorted(requeue_values - _REQUEUE_VALID_VERDICTS)
    if invalid:
        raise BadParameter(
            f"--requeue: invalid verdict(s) {invalid}; "
            f"valid values are {sorted(_REQUEUE_VALID_VERDICTS)}"
        )
    existing = _existing_pair_count(out)
    if existing > 0 and not rebuild and not append:
        echo(
            f"review.db at {out} already contains {existing} pairs. "
            f"Pass --rebuild to drop and recreate, or --append to add to it.",
            err=True,
        )
        raise Exit(code=2)
    if rebuild and out.exists():
        out.unlink()
    resolved_log_file = _configure_logging("build-queue", log_file)
    resolved_budget = default_budget() if budget is None else scale_budget(default_budget(), budget)
    summary = build_queue(
        pool=pool,
        index_path=index,
        out_path=out,
        vault_path=vault,
        budget=resolved_budget,
        matching_config=_load_default_matching_config(),
        pairing_config=_load_default_pairing_config(),
        seed=seed,
        workers=workers,
        sample_per_lang=sample_per_lang,
        verbosity=verbose,
        log_file=resolved_log_file,
        requeue_verdicts=requeue_values,
    )
    strata = " ".join(f"{label}={count}" for label, count in sorted(summary.stratum_counts.items()))
    echo(
        f"records_sampled={summary.records_sampled} "
        f"records_matched={summary.records_matched} "
        f"pairs_written={summary.pairs_written} "
        f"strata=[{strata}]"
    )


@app.command(name="review")
def review_command(
    db: Annotated[
        Path, Option("--db", help="SQLite review database produced by `build-queue`.")
    ] = _DEFAULT_REVIEW_DB_PATH,
    vault: Annotated[
        Path,
        Option(
            "--vault",
            help="JSONL label vault; each accepted verdict is appended here.",
        ),
    ] = _DEFAULT_VAULT_PATH,
    host: Annotated[
        str, Option("--host", help="Interface to bind the local review server.")
    ] = _DEFAULT_REVIEW_HOST,
    port: Annotated[int, Option("--port", help="Port for the local review server.")] = (
        _DEFAULT_REVIEW_PORT
    ),
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Launch the local keyboard-driven review UI over a review database."""
    _configure_logging("review", log_file)
    echo(f"serving review UI for {db} (vault: {vault}) at http://{host}:{port}")
    serve(db, vault, host=host, port=port)


@app.command(name="seed-vault")
def seed_vault_command(
    db: Annotated[
        Path,
        Option("--db", help="SQLite review database whose current labels to dump."),
    ] = _DEFAULT_REVIEW_DB_PATH,
    vault: Annotated[
        Path,
        Option("--vault", help="JSONL label vault to upsert into (created if absent)."),
    ] = _DEFAULT_VAULT_PATH,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """One-shot migration: dump every current label from ``--db`` into ``--vault``.

    Idempotent: entries already present at the same ``labeled_at`` for the
    same pair are skipped. A different ``labeled_at`` upserts the entry in
    place, since the vault holds exactly one entry per pair.
    """
    _configure_logging("seed-vault", log_file)
    existing = current_entries(vault)
    seeded = 0
    skipped = 0
    with ReviewDb.connect(db) as connection:
        for label in connection.iter_current_labels():
            key = (label.marc_control_id, label.nypl_uuid)
            present = existing.get(key)
            if present is not None and present.labeled_at == label.labeled_at:
                skipped += 1
                continue
            marc = json_decode(label.marc_json.encode("utf-8"), type=MarcRecord)
            entry = VaultEntry(
                schema=SCHEMA_VERSION,
                marc_control_id=label.marc_control_id,
                nypl_uuid=label.nypl_uuid,
                verdict=label.verdict,
                note=label.note,
                labeled_at=label.labeled_at,
                labeler=_LABELER,
                marc_identifiers=extract_marc_identifiers(marc),
                cce_regnum=label.cce_regnum,
                cce_renewal_id=label.cce_renewal_id,
                cce_renewal_oreg=label.cce_renewal_oreg,
                categories=(),
            )
            upsert_entry(vault, entry)
            seeded += 1
    echo(f"seeded {seeded} labels; skipped {skipped} already-present")


@app.command(name="dump-vault-marcs")
def dump_vault_marcs_command(
    vault: Annotated[
        Path,
        Option("--vault", help="JSONL label vault whose MARCs to dump."),
    ] = _DEFAULT_VAULT_PATH,
    pool: Annotated[
        Path,
        Option("--pool", help="Root dir whose <lang>/*.xml shards form the candidate pool."),
    ] = _DEFAULT_POOL_PATH,
    out: Annotated[
        Path,
        Option("--out", help="Destination MARCXML file."),
    ] = _DEFAULT_VAULT_MARCS_PATH,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Write a MARCXML collection of every vault MARC for downstream publication."""
    _configure_logging("dump-vault-marcs", log_file)
    report = dump_vault_marcs(vault, pool, out)
    echo(
        f"wrote {report.marcs_written} records to {out} "
        f"(vault_entries={report.vault_entries} "
        f"distinct_marcs={report.distinct_marcs_requested} "
        f"missing={report.marcs_missing})"
    )


@app.command(name="publish-linkage")
def publish_linkage_command(
    vault: Annotated[
        Path,
        Option("--vault", help="JSONL label vault to reshape for publication."),
    ] = _DEFAULT_VAULT_PATH,
    training_out: Annotated[
        Path,
        Option(
            "--training-out",
            help="Destination JSONL with every adjudicated verdict (match / no_match / unsure).",
        ),
    ] = _DEFAULT_PUBLISHED_TRAINING_PATH,
    matches_out: Annotated[
        Path,
        Option("--matches-out", help="Destination JSONL with match rows only."),
    ] = _DEFAULT_PUBLISHED_MATCHES_PATH,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Reshape the vault into the published JSONL pair (training + matches-only)."""
    _configure_logging("publish-linkage", log_file)
    report = publish_linkage(vault, training_out, matches_out)
    echo(
        f"wrote {report.rows_written} rows to {training_out} "
        f"({report.matches} matches also written to {matches_out}; "
        f"no_matches={report.no_matches} unsures={report.unsures})"
    )


@app.command(name="vault-into-queue")
def vault_into_queue_command(
    db: Annotated[
        Path, Option("--db", help="Existing SQLite review database to backfill.")
    ] = _DEFAULT_REVIEW_DB_PATH,
    vault: Annotated[
        Path,
        Option(
            "--vault",
            help="JSONL label vault whose entries are checked against the queue.",
        ),
    ] = _DEFAULT_VAULT_PATH,
    pool: Annotated[
        Path,
        Option(
            "--pool",
            help="Root dir whose <lang>/*.xml shards form the candidate pool.",
        ),
    ] = _DEFAULT_POOL_PATH,
    index: Annotated[
        Path, Option("--index", help="LMDB env produced by `pd-matcher index build`.")
    ] = _DEFAULT_INDEX_PATH,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Backfill vault-labeled pairs that are missing from ``--db``.

    Tactical bridge until ``build-queue`` always includes vault MARCs
    (``jpstroop/pd-matcher#33``). Reads the vault, finds every
    ``(marc_control_id, nypl_uuid)`` not present in ``--db``, locates the MARC
    in ``--pool`` and the CCE registration in ``--index``, scores the specific
    pair with the matcher, and inserts both the ``review_pair`` row and the
    pre-existing vault verdict.
    """
    _configure_logging("vault-into-queue", log_file)
    summary = vault_into_queue(
        db_path=db,
        vault_path=vault,
        pool_path=pool,
        index_path=index,
        matching_config=_load_default_matching_config(),
        pairing_config=_load_default_pairing_config(),
    )
    echo(
        f"backfilled {summary.backfilled} vault pairs; "
        f"{summary.missing_in_pool} MARC records not found in pool; "
        f"{summary.missing_in_index} CCE records not found in index; "
        f"{summary.already_present} already present (skipped)"
    )


@app.command(name="migrate-vault-v3")
def migrate_vault_v3_command(
    vault: Annotated[
        Path,
        Option("--vault", help="JSONL label vault to migrate in place."),
    ] = _DEFAULT_VAULT_PATH,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Fold pre-schema-3 ``reasons`` / ``field_annotations`` into the note text.

    Archives the original vault to ``<vault>.pre-v3`` and rewrites the
    canonical path with one schema-3 :class:`VaultEntry` per line. Idempotent:
    re-running on an already-migrated vault is a no-op and creates no
    additional archive.
    """
    _configure_logging("migrate-vault-v3", log_file)
    report = migrate_vault_v3(vault)
    archive_note = (
        f"archived original to {report.archive_path}"
        if report.archive_path is not None
        else "no migration needed (vault already at schema 3 or empty)"
    )
    echo(
        f"migrated {report.total_entries} entries; "
        f"folded reasons on {report.reasons_folded}; "
        f"folded annotations on {report.annotations_folded}; "
        f"{archive_note}"
    )


@app.command(name="migrate-vault-v4")
def migrate_vault_v4_command(
    vault: Annotated[
        Path,
        Option("--vault", help="JSONL label vault to migrate in place."),
    ] = _DEFAULT_VAULT_PATH,
    index: Annotated[
        Path,
        Option("--index", help="LMDB env produced by `pd-matcher index build`."),
    ] = _DEFAULT_INDEX_PATH,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Backfill schema-4 CCE-side identifier fields onto every vault entry.

    Looks each entry's ``nypl_uuid`` up in the CCE index at ``--index`` and
    copies ``regnum`` / ``renewal_id`` / ``renewal_oreg`` into three flat
    top-level fields on the entry, bumping the ``schema`` to 4. Entries whose
    UUID no longer resolves (data drift) keep ``None`` for the new fields but
    still get ``schema=4`` and are counted as orphaned. The pre-migration
    vault lives in git history; no on-disk archive is written. Idempotent:
    re-running on an already-migrated vault is a no-op and does not rewrite
    the file.
    """
    _configure_logging("migrate-vault-v4", log_file)
    if not vault.exists() or vault.stat().st_size == 0:
        report = migrate_vault_v4(vault, lambda _uuid: None)
    else:
        with NyplIndexLookup(index) as lookup:
            report = migrate_vault_v4(vault, lookup.get_registration)
    echo(
        f"migrated {report.total_entries} entries; "
        f"enriched {report.enriched}; "
        f"orphaned {report.orphaned}"
    )


@app.command(name="migrate-vault-v5")
def migrate_vault_v5_command(
    vault: Annotated[
        Path,
        Option("--vault", help="JSONL label vault to migrate in place."),
    ] = _DEFAULT_VAULT_PATH,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Bump every vault entry to schema 5 and backfill ``categories`` with ``[]``.

    Schema 5 adds the ``categories`` field to capture recurring rationale
    patterns (series-vs-volume mismatches, translations, OCR confusion, etc.)
    as structured data rather than free-text notes. The migration is
    uniform: every pre-v5 entry gets an empty tuple. The pre-migration vault
    lives in git history; no on-disk archive is written. Idempotent:
    re-running on an already-migrated vault is a no-op and does not rewrite
    the file.
    """
    _configure_logging("migrate-vault-v5", log_file)
    report = migrate_vault_v5(vault)
    echo(f"migrated {report.total_entries} entries; bumped {report.migrated} to schema 5")
