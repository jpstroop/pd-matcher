"""Typer entry point for the ground-truth acquisition tool."""

from datetime import date
from logging import INFO
from logging import basicConfig
from pathlib import Path
from typing import Annotated

from msgspec.json import decode as json_decode
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.models import MarcRecord
from typer import Option
from typer import Typer
from typer import echo

from pd_groundtruth.acquire import acquire
from pd_groundtruth.acquire import default_min_year
from pd_groundtruth.build_queue import build_queue
from pd_groundtruth.build_queue import load_default_ruleset
from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import append_entry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.label_vault import extract_marc_identifiers
from pd_groundtruth.manifest import DEFAULT_MANIFEST_URL
from pd_groundtruth.review.server import serve
from pd_groundtruth.review_db import ReviewDb
from pd_groundtruth.sampling import default_budget
from pd_groundtruth.sampling import scale_budget

app = Typer(add_completion=False, help="Acquire Princeton MARC ground-truth candidates.")

_DEFAULT_PER_DECADE_CAP = 20000
_DEFAULT_SEED = 42
_DEFAULT_WORKERS = 8
_DEFAULT_SAMPLE_PER_LANG = 1500
_DEFAULT_REVIEW_HOST = "127.0.0.1"
_DEFAULT_REVIEW_PORT = 8000
_DEFAULT_VAULT_PATH = Path("label_vault.jsonl")
_LABELER = "jpstroop"


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
) -> None:
    """Stream dumps and write eligible records as per-language MARCXML shards."""
    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
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


@app.command(name="build-queue")
def build_queue_command(
    pool: Annotated[
        Path,
        Option("--pool", help="Root dir whose <lang>/*.xml shards form the candidate pool."),
    ],
    index: Annotated[
        Path, Option("--index", help="LMDB env produced by `pd-matcher index build`.")
    ],
    out: Annotated[Path, Option("--out", help="Destination SQLite review database.")],
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
) -> None:
    """Match a stratified pool sample and write a SQLite review queue."""
    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    resolved_budget = default_budget() if budget is None else scale_budget(default_budget(), budget)
    summary = build_queue(
        pool=pool,
        index_path=index,
        out_path=out,
        vault_path=vault,
        budget=resolved_budget,
        matching_config=_load_default_matching_config(),
        pairing_config=_load_default_pairing_config(),
        ruleset=load_default_ruleset(),
        copyright_config=CopyrightAssessmentConfig(as_of_year=date.today().year),
        seed=seed,
        workers=workers,
        sample_per_lang=sample_per_lang,
        verbosity=verbose,
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
    db: Annotated[Path, Option("--db", help="SQLite review database produced by `build-queue`.")],
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
) -> None:
    """Launch the local keyboard-driven review UI over a review database."""
    echo(f"serving review UI for {db} (vault: {vault}) at http://{host}:{port}")
    serve(db, vault, host=host, port=port)


@app.command(name="seed-vault")
def seed_vault_command(
    db: Annotated[
        Path,
        Option("--db", help="SQLite review database whose current labels to dump."),
    ],
    vault: Annotated[
        Path,
        Option("--vault", help="JSONL label vault to append into (created if absent)."),
    ] = _DEFAULT_VAULT_PATH,
) -> None:
    """One-shot migration: dump every current label from ``--db`` into ``--vault``.

    Idempotent: entries already present (matching ``marc_control_id`` +
    ``nypl_uuid`` + ``labeled_at``) are skipped. Different ``labeled_at`` for
    the same pair is treated as a new event and appended.
    """
    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
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
                reasons=label.reasons,
                note=label.note,
                labeled_at=label.labeled_at,
                labeler=_LABELER,
                marc_identifiers=extract_marc_identifiers(marc),
            )
            append_entry(vault, entry)
            seeded += 1
    echo(f"seeded {seeded} labels; skipped {skipped} already-present")
