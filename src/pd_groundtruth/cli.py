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

from pd_groundtruth.acquire import _DEFAULT_MIN_FREE_SPACE_MB
from pd_groundtruth.acquire import acquire
from pd_groundtruth.acquire import default_min_year
from pd_groundtruth.active_learning import ActiveLearningSummary
from pd_groundtruth.active_learning import run_active_learning
from pd_groundtruth.active_select import DEFAULT_LANGUAGE_WEIGHTS
from pd_groundtruth.build_corpus import build_corpus
from pd_groundtruth.build_queue import build_queue
from pd_groundtruth.build_renewal_queue import build_renewal_queue
from pd_groundtruth.disk_guard import InsufficientDiskSpaceError
from pd_groundtruth.dump_vault_marcs import dump_vault_marcs
from pd_groundtruth.enrich_vault import run_enrich
from pd_groundtruth.filter import filter_marcxml
from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.label_vault import extract_marc_identifiers
from pd_groundtruth.label_vault import upsert_entry
from pd_groundtruth.manifest import DEFAULT_MANIFEST_URL
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
from pd_groundtruth.vault_migration import migrate_vault_v6
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.cli import _ScorerChoice
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.models import MarcRecord

app = Typer(add_completion=False, help="Acquire Princeton MARC ground-truth candidates.")

_DEFAULT_PER_DECADE_CAP = 20000
_DEFAULT_SEED = 42
_DEFAULT_WORKERS = 8
_DEFAULT_SAMPLE_PER_LANG = 1500
_DEFAULT_RENEWAL_MIN_SCORE = 60.0
_DEFAULT_REG_MIN_SCORE = 90.0
_DEFAULT_REG_SCORER = _ScorerChoice.LEARNED
_DEFAULT_REVIEW_HOST = "127.0.0.1"
_DEFAULT_REVIEW_PORT = 8000
_DEFAULT_POOL_PATH = Path("data/candidates")
_DEFAULT_INDEX_PATH = Path("caches/cce.lmdb")
_DEFAULT_REVIEW_DB_PATH = Path("data/review.db")
_DEFAULT_ACTIVE_DB_PATH = Path("data/active_learning.db")
_DEFAULT_ACTIVE_TARGET = 1000
_DEFAULT_VAULT_PATH = Path("data/training/label_vault.jsonl")
_DEFAULT_VAULT_MARCS_PATH = Path("data/training/marc.xml")
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
    min_free_space_mb: Annotated[
        int,
        Option(
            "--min-free-space-mb",
            help=(
                "Abort safely if the temp-download or output filesystem drops below this "
                "many MB free, checked before and during each download. The partial "
                "per-language shards are finalized as valid <collection> files. "
                "0 disables the guard."
            ),
        ),
    ] = _DEFAULT_MIN_FREE_SPACE_MB,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Stream dumps and write eligible records as per-language MARCXML shards.

    If free disk space drops below ``--min-free-space-mb`` the run aborts safely:
    the partial per-language shards are finalized as valid ``<collection>`` files
    and the command exits non-zero.
    """
    _configure_logging("acquire", log_file)
    resolved_min_year = default_min_year() if min_year is None else min_year
    try:
        report = acquire(
            out_dir=out_dir,
            per_decade_cap=per_decade_cap,
            min_year=resolved_min_year,
            manifest_url=manifest_url,
            max_dumps=max_dumps,
            min_free_space_mb=min_free_space_mb,
        )
    except InsufficientDiskSpaceError as error:
        echo(
            f"aborted: {error} "
            f"(wrote {error.records_written} records across {error.dumps_written} dumps "
            f"to {out_dir}; threshold was {min_free_space_mb} MB)",
            err=True,
        )
        raise Exit(code=1) from error
    kept = " ".join(f"{language}={count}" for language, count in report.kept_by_language.items())
    echo(
        f"dumps_processed={report.dumps_processed} "
        f"records_scanned={report.records_scanned} "
        f"kept_by_language=[{kept}] "
        f"shards_written={report.shards_written} "
        f"stopped_reason={report.stopped_reason}"
    )


def _parse_languages(value: str | None) -> frozenset[str] | None:
    """Parse a comma-separated ``--languages`` value into a code set.

    ``None`` keeps every record that passes eligibility (any supported
    language). A value such as ``"eng,fre"`` restricts output to those 008
    codes. Whitespace around each code is stripped.

    Raises:
        BadParameter: When the value is present but contains no non-empty code.
    """
    if value is None:
        return None
    codes = frozenset(code.strip() for code in value.split(",") if code.strip())
    if not codes:
        raise BadParameter("--languages must list at least one language code")
    return codes


@app.command(name="filter")
def filter_command(
    input_path: Annotated[
        Path,
        Option("--input", help="Plain MARCXML file to filter (the format `match --marc` reads)."),
    ],
    output_path: Annotated[
        Path,
        Option("--output", help="Destination MARCXML <collection> of the eligible records."),
    ],
    min_year: Annotated[
        int | None,
        Option(
            "--min-year",
            help="Lower bound for publication year (the moving wall, today.year - 95).",
        ),
    ] = None,
    languages: Annotated[
        str | None,
        Option(
            "--languages",
            help=(
                "Comma-separated 008 codes to keep (e.g. 'eng,fre'). "
                "Default keeps every eligible record (eng, fre, ger, spa, ita)."
            ),
        ),
    ] = None,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Write only the production-eligible records of a MARCXML file, uncapped.

    Applies the same eligibility logic as ``acquire`` (monograph, not an
    electronic resource, publication year within the moving wall .. 1977, a
    supported language, a title) to every ``<record>`` in ``--input`` and writes
    the survivors to a single MARCXML ``<collection>`` at ``--output`` that
    ``pd-matcher match --marc`` can consume. Unlike ``acquire`` there is no
    per-bucket cap. By default every eligible record is kept regardless of which
    supported language it is in; pass ``--languages`` to narrow the output to a
    subset.
    """
    _configure_logging("filter", log_file)
    resolved_min_year = default_min_year() if min_year is None else min_year
    report = filter_marcxml(
        input_path=input_path,
        output_path=output_path,
        min_year=resolved_min_year,
        languages=_parse_languages(languages),
    )
    breakdown = " ".join(
        f"{reason}={count}" for reason, count in sorted(report.dropped_by_reason.items())
    )
    echo(
        f"scanned={report.scanned} kept={report.kept} dropped={report.dropped} "
        f"dropped_by_reason=[{breakdown}]"
    )


@app.command(name="build-corpus")
def build_corpus_command(
    output_path: Annotated[
        Path,
        Option("--output", help="Destination MARCXML <collection> of the in-scope corpus."),
    ],
    min_year: Annotated[
        int | None,
        Option(
            "--min-year",
            help="Lower bound for publication year (the moving wall, today.year - 95).",
        ),
    ] = None,
    languages: Annotated[
        str | None,
        Option(
            "--languages",
            help=(
                "Comma-separated 008 codes to keep (e.g. 'eng,fre'). "
                "Default keeps every eligible record (eng, fre, ger, spa, ita)."
            ),
        ),
    ] = None,
    manifest_url: Annotated[
        str, Option("--manifest-url", help="Dump manifest JSON URL.")
    ] = DEFAULT_MANIFEST_URL,
    max_dumps: Annotated[
        int | None, Option("--max-dumps", help="Cap the number of dumps processed.")
    ] = None,
    min_free_space_mb: Annotated[
        int,
        Option(
            "--min-free-space-mb",
            help=(
                "Abort safely if the temp-download or output filesystem drops below this "
                "many MB free, checked before and during each download. The partial corpus "
                "is finalized as a valid <collection>. 0 disables the guard."
            ),
        ),
    ] = _DEFAULT_MIN_FREE_SPACE_MB,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Stream the whole catalog and write the full in-scope matching corpus.

    Downloads every dump in the manifest, applies the same eligibility logic as
    ``acquire`` / ``filter`` (monograph, not an electronic resource, publication
    year within the moving wall .. 1977, a supported language, a title) to every
    ``<record>``, and writes the survivors to a single MARCXML ``<collection>``
    at ``--output`` that ``pd-matcher match --marc`` can consume. Unlike
    ``acquire`` there is no per-(language, decade) cap: this is the uncapped
    production corpus extractor, not the capped training-set sampler. Each dump
    is downloaded to a temp file, scanned, and deleted before the next one, so
    the raw catalog never accumulates on disk.

    If free disk space drops below ``--min-free-space-mb`` the run aborts safely:
    the partial corpus is finalized as a valid ``<collection>`` and the command
    exits non-zero.
    """
    _configure_logging("build-corpus", log_file)
    resolved_min_year = default_min_year() if min_year is None else min_year
    try:
        report = build_corpus(
            output_path=output_path,
            min_year=resolved_min_year,
            languages=_parse_languages(languages),
            manifest_url=manifest_url,
            max_dumps=max_dumps,
            min_free_space_mb=min_free_space_mb,
        )
    except InsufficientDiskSpaceError as error:
        echo(
            f"aborted: {error} "
            f"(wrote {error.records_written} records across {error.dumps_written} dumps "
            f"to {output_path}; threshold was {min_free_space_mb} MB)",
            err=True,
        )
        raise Exit(code=1) from error
    breakdown = " ".join(
        f"{reason}={count}" for reason, count in sorted(report.dropped_by_reason.items())
    )
    echo(
        f"dumps_processed={report.dumps_processed} records_scanned={report.records_scanned} "
        f"kept={report.kept} dropped={report.dropped} dropped_by_reason=[{breakdown}]"
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


@app.command(name="build-renewal-queue")
def build_renewal_queue_command(
    pool: Annotated[
        Path,
        Option("--pool", help="Root dir whose <lang>/*.xml shards form the candidate pool."),
    ] = _DEFAULT_POOL_PATH,
    index: Annotated[
        Path, Option("--index", help="LMDB env produced by `pd-matcher index build`.")
    ] = _DEFAULT_INDEX_PATH,
    out: Annotated[
        Path, Option("--out", help="Destination SQLite review database (appended to).")
    ] = _DEFAULT_REVIEW_DB_PATH,
    min_score: Annotated[
        float,
        Option(
            "--min-score",
            help=(
                "Renewal-arm score floor (0-100); a MARC whose best renewal scores below it "
                "is not a renewal-haver and is skipped before the registration check runs."
            ),
        ),
    ] = _DEFAULT_RENEWAL_MIN_SCORE,
    reg_min_score: Annotated[
        float,
        Option(
            "--reg-min-score",
            help=(
                "Registration-arm score floor (0-100); a registration at or above it in the "
                "renewal's odat year excludes the book (scenario 2/3, not scenario 4)."
            ),
        ),
    ] = _DEFAULT_REG_MIN_SCORE,
    reg_scorer: Annotated[
        _ScorerChoice,
        Option(
            "--reg-scorer",
            help="Combiner for the registration check (weighted_mean|learned).",
        ),
    ] = _DEFAULT_REG_SCORER,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Queue scenario-4 (renewal-only) MARC↔renewal pairs for labeling.

    Renewal-first: for every pool MARC not already in the ``--out`` review DB the
    cheap renewal search runs first, keeping only books whose best renewal clears
    ``--min-score`` (the renewal-havers). Each renewal-haver is then checked for a
    registration in the renewal's original-registration (``odat``) year using
    ``--reg-scorer`` and the ``--reg-min-score`` floor:

    * a registration at or above the floor excludes the book (scenario 2/3 — a
      registration exists);
    * no registration in the ``odat`` year emits the renewal as a scenario-4
      ``pairing_type="renewal"`` pair with an audit note.

    The expensive registration check only ever runs for renewal-havers, and only
    within the single ``odat`` year. Existing registration pairs are left
    untouched and never duplicated.
    """
    _configure_logging("build-renewal-queue", log_file)
    summary = build_renewal_queue(
        pool=pool,
        index_path=index,
        out_path=out,
        matching_config=_load_default_matching_config(),
        pairing_config=_load_default_pairing_config(),
        min_score=min_score,
        reg_min_score=reg_min_score,
        reg_scorer=reg_scorer.value,
    )
    echo(
        f"records_scanned={summary.records_scanned} "
        f"renewal_havers={summary.renewal_havers} "
        f"reg_excluded={summary.reg_excluded} "
        f"scenario4_written={summary.scenario4_written}"
    )


def _parse_language_weights(values: list[str] | None) -> dict[str, float]:
    """Parse repeated ``lang=weight`` options into a weights mapping.

    ``None`` or an empty list returns the English-heavy default
    (:data:`pd_groundtruth.active_select.DEFAULT_LANGUAGE_WEIGHTS`). Each value
    is ``"<lang>=<weight>"`` (e.g. ``"eng=0.7"``); duplicate languages take the
    last value.

    Raises:
        BadParameter: On a malformed token or a non-positive / unparseable
            weight.
    """
    if not values:
        return dict(DEFAULT_LANGUAGE_WEIGHTS)
    weights: dict[str, float] = {}
    for token in values:
        language, separator, raw = token.partition("=")
        if not separator or not language:
            raise BadParameter(f"--weight expects 'lang=weight', got {token!r}")
        try:
            weight = float(raw)
        except ValueError as exc:
            raise BadParameter(f"--weight: {raw!r} is not a number") from exc
        if weight <= 0.0:
            raise BadParameter(f"--weight: {language} weight must be positive, got {weight!r}")
        weights[language] = weight
    return weights


def _echo_active_summary(summary: ActiveLearningSummary, out: Path) -> None:
    """Print the per-bucket distribution + disagreement spread for one run."""
    mode = "(dry-run) " if summary.dry_run else ""
    plans = " ".join(
        f"{plan.language}={plan.selected}/{plan.target}" for plan in summary.language_plans
    )
    echo(
        f"{mode}selected={summary.selected} excluded_in_vault={summary.excluded} "
        f"out_of_scope={summary.out_of_scope} scored={summary.scored} "
        f"by_language=[{plans}]"
    )
    echo(f"{'bucket':<14} {'count':>6} {'min':>7} {'mean':>7} {'max':>7}")
    echo("-" * 44)
    for stats in summary.buckets:
        echo(
            f"{stats.bucket:<14} {stats.count:>6} "
            f"{stats.min_disagreement:>7.3f} {stats.mean_disagreement:>7.3f} "
            f"{stats.max_disagreement:>7.3f}"
        )
    if summary.dry_run:
        echo(f"dry-run: {summary.informative()} informative pairs would be written to {out}")
    else:
        echo(f"wrote {summary.written} informative pairs to {out}")


@app.command(name="build-active-queue")
def build_active_queue_command(
    pool: Annotated[
        Path,
        Option("--pool", help="Root dir whose <lang>/*.xml shards form the candidate pool."),
    ] = _DEFAULT_POOL_PATH,
    index: Annotated[
        Path, Option("--index", help="LMDB env produced by `pd-matcher index build`.")
    ] = _DEFAULT_INDEX_PATH,
    out: Annotated[
        Path, Option("--out", help="Destination SQLite review DB for the informative pairs.")
    ] = _DEFAULT_ACTIVE_DB_PATH,
    vault: Annotated[
        Path,
        Option(
            "--vault",
            help="JSONL label vault; its MARCs are excluded from selection.",
        ),
    ] = _DEFAULT_VAULT_PATH,
    target: Annotated[
        int,
        Option("--target", help="Number of unseen records to select across all languages."),
    ] = _DEFAULT_ACTIVE_TARGET,
    weight: Annotated[
        list[str] | None,
        Option(
            "--weight",
            help=(
                "Language selection weight as 'lang=weight' (repeatable). "
                "Default is English-heavy: eng=0.70, fre/ger/ita/spa=0.075 each."
            ),
        ),
    ] = None,
    seed: Annotated[int, Option("--seed", help="Seed for the reservoir samplers.")] = _DEFAULT_SEED,
    dry_run: Annotated[
        bool,
        Option(
            "--dry-run",
            help="Print the per-bucket distribution but write no DB (preview a run).",
        ),
    ] = False,
    rebuild: Annotated[
        bool,
        Option("--rebuild", help="Delete the target --out database before writing (destructive)."),
    ] = False,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Select unseen records, dual-score them, and queue matcher disagreements.

    Active-learning sibling of ``build-queue`` (issue #81): instead of a
    stratified sweep, it selects ~``--target`` MARC records NOT already in the
    vault, scores each with BOTH the weighted-mean and learned matchers over one
    shared Evidence stream, buckets them by matcher-vs-matcher disagreement, and
    writes the ``informative`` pairs (committee splits, large same-pick gaps,
    both-uncertain) into ``--out`` for review via ``pd-groundtruth review``.

    The learned model is required; if its artifact or ``lightgbm`` is missing
    the run aborts naming ``train-scorer`` / ``pdm install --group ml``. Pass
    ``--dry-run`` to preview the bucket distribution without writing.
    """
    if not dry_run and not rebuild and out.exists():
        echo(
            f"active-learning DB at {out} already exists. "
            f"Pass --rebuild to drop and recreate, or --dry-run to preview only.",
            err=True,
        )
        raise Exit(code=2)
    if not dry_run and rebuild and out.exists():
        out.unlink()
    _configure_logging("build-active-queue", log_file)
    weights = _parse_language_weights(weight)
    summary = run_active_learning(
        pool=pool,
        index_path=index,
        out_path=out,
        vault_path=vault,
        matching_config=_load_default_matching_config(),
        pairing_config=_load_default_pairing_config(),
        weights=weights,
        target=target,
        seed=seed,
        dry_run=dry_run,
    )
    _echo_active_summary(summary, out)


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


@app.command(name="migrate-vault-v6")
def migrate_vault_v6_command(
    vault: Annotated[
        Path,
        Option("--vault", help="JSONL label vault to migrate in place."),
    ] = _DEFAULT_VAULT_PATH,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Bump every vault entry to schema 7 and backfill ``match_source``.

    Schema 7 adds the ``match_source`` field recording which CCE pathway
    surfaced the pair. Every pre-schema-7 label came from the registration
    pathway, so the migration uniformly stamps ``"registration"`` on each.
    The pre-migration vault lives in git history; no on-disk archive is
    written. Idempotent: re-running on an already-migrated vault is a no-op
    and does not rewrite the file.
    """
    _configure_logging("migrate-vault-v6", log_file)
    report = migrate_vault_v6(vault)
    echo(f"migrated {report.total_entries} entries; bumped {report.migrated} to schema 7")


@app.command(name="enrich-vault")
def enrich_vault_command(
    vault: Annotated[
        Path,
        Option("--vault", help="JSONL label vault to enrich in place."),
    ] = _DEFAULT_VAULT_PATH,
    index: Annotated[
        Path, Option("--index", help="LMDB env produced by `pd-matcher index build`.")
    ] = _DEFAULT_INDEX_PATH,
    marc_collection: Annotated[
        Path,
        Option(
            "--marc-collection",
            help="Single MARCXML <collection> of vault MARCs (used when --pool is absent).",
        ),
    ] = _DEFAULT_VAULT_MARCS_PATH,
    pool: Annotated[
        Path | None,
        Option(
            "--pool",
            help=(
                "Root dir whose <lang>/*.xml shards form the candidate pool. "
                "Mutually exclusive with --marc-collection."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        Option("--dry-run", help="Compute and report the enrichment without writing the vault."),
    ] = False,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Backfill schema-6 machine-derived fields onto every vault entry.

    For each entry, resolves the MARC and CCE registration, copies the CCE-side
    ``reg_year`` / ``was_renewed`` / ``renewal_year``, scores the pair through
    both matcher combiners over one shared Evidence stream, and stamps the
    producing ``matcher_version``. Human-entered fields are preserved verbatim;
    the enriched vault is written atomically at schema 6. Pass ``--dry-run`` to
    report the counts without writing. By default the MARCs are read from the
    committed training collection (``--marc-collection``); pass ``--pool`` to
    read from a sharded acquired pool instead.
    """
    _configure_logging("enrich-vault", log_file)
    report = run_enrich(
        vault_path=vault,
        index_path=index,
        matching_config=_load_default_matching_config(),
        pairing_config=_load_default_pairing_config(),
        pool_path=pool,
        marc_collection_path=None if pool is not None else marc_collection,
        dry_run=dry_run,
    )
    mode = "(dry-run) " if dry_run else ""
    echo(
        f"{mode}enriched {report.enriched}/{report.total_entries} entries; "
        f"learned_scored={report.learned_scored}; "
        f"{report.missing_in_pool} MARC records not found in pool; "
        f"{report.missing_in_index} CCE records not found in index"
    )
