"""Typer entry point for the ``pd-matcher`` command.

Phase 7 wires every subcommand to its implementation module. Errors are
reported on stderr with these exit codes:

* ``0`` — success
* ``1`` — runtime error (missing file, bad config, IO failure)
* ``2`` — argument error (typer-default for parse failures, plus the
  validators and the not-yet-implemented stubs)
* ``130`` — interrupted by SIGINT
"""

from datetime import UTC
from datetime import datetime
from importlib.resources import as_file
from importlib.resources import files
from json import dumps
from logging import WARNING
from pathlib import Path
from typing import Annotated

from msgspec import DecodeError
from msgspec import to_builtins
from typer import BadParameter
from typer import Exit
from typer import Option
from typer import Typer
from typer import echo

from pd_matcher.config.loader import ConfigError
from pd_matcher.config.loader import load_matching_config
from pd_matcher.config.loader import load_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.eval.ground_truth import EvalReport
from pd_matcher.eval.ground_truth import run_eval
from pd_matcher.eval.metrics import ThresholdPoint
from pd_matcher.index.builder import BuildReport
from pd_matcher.index.builder import build_index
from pd_matcher.index.lookup import IndexStats
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.logging_config import configure_logging
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.calibrator import load_calibrator
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.prepare import PrepareReport
from pd_matcher.match.prepare import prepare_marc
from pd_matcher.match.prepare import read_manifest
from pd_matcher.workers import RunReport
from pd_matcher.workers import run_match

_ARG_ERROR_EXIT_CODE: int = 2
_RUNTIME_ERROR_EXIT_CODE: int = 1
_INTERRUPTED_EXIT_CODE: int = 130

_IDF_CACHE_NAME: str = "idf.msgpack"
_CALIBRATOR_NAME: str = "calibrator.msgpack"

_YEAR_WINDOW_MIN: int = 0
_YEAR_WINDOW_MAX: int = 100

_DEFAULT_VAULT_PATH: Path = Path("data/label_vault.jsonl")
_DEFAULT_POOL_PATH: Path = Path("data/candidates")
_SWEEP_PREVIEW_STEP: int = 4


class _LogSettings:
    """Process-wide log settings captured by the root callback.

    Spawned match workers reconfigure logging from scratch, so they need the
    level and JSON flag the user chose at the root; typer does not thread
    callback values into subcommands, so they are stashed here on the one
    instance the callback writes and the subcommands read.
    """

    __slots__ = ("json_logs", "level")

    def __init__(self) -> None:
        self.level: str = "INFO"
        self.json_logs: bool = False


_LOG_SETTINGS: _LogSettings = _LogSettings()
_LOG_DIR_NAME: str = "logs"


def _utc_timestamp() -> str:
    """Return a filename-safe UTC timestamp (e.g. ``20260523-004530``)."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _resolve_log_file(command: str, override: Path | None) -> Path:
    """Pick the log file path for ``command`` honoring ``--log-file``.

    Args:
        command: Command name embedded in the auto-generated filename.
        override: Explicit path from ``--log-file``; when ``None`` the path
            is auto-generated under ``logs/`` relative to CWD.

    Returns:
        The resolved path. The parent directory is *not* created here;
        :func:`configure_logging` does that just before opening the file.
    """
    if override is not None:
        return override
    return Path(_LOG_DIR_NAME) / f"{command}_{_utc_timestamp()}.log"


app: Typer = Typer(
    help="MARC ↔ NYPL CCE linkage matcher.",
    no_args_is_help=True,
)
index_app: Typer = Typer(
    help="Build and inspect the LMDB NYPL index.",
    no_args_is_help=True,
)
app.add_typer(index_app, name="index")


@app.callback()
def _main(
    log_level: Annotated[
        str,
        Option("--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR)."),
    ] = "INFO",
    json_logs: Annotated[
        bool,
        Option(
            "--json-logs/--no-json-logs",
            help="Emit logs as JSON (one object per line).",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        Option(
            "--quiet/--no-quiet",
            help="Suppress everything below WARNING (overrides --log-level).",
        ),
    ] = False,
) -> None:
    """Initialize logging before any subcommand runs."""
    effective_level = "WARNING" if quiet else log_level
    configure_logging(level=effective_level, json_output=json_logs)
    _LOG_SETTINGS.level = effective_level
    _LOG_SETTINGS.json_logs = json_logs
    if quiet:
        from logging import getLogger

        getLogger().setLevel(WARNING)


def _enable_log_file(command: str, override: Path | None) -> Path:
    """Resolve and attach the per-invocation log file, echo it to stderr.

    Reconfigures logging with the resolved file path and echoes a short
    notice so users see where the run's output is being persisted.
    """
    path = _resolve_log_file(command, override)
    configure_logging(
        level=_LOG_SETTINGS.level,
        json_output=_LOG_SETTINGS.json_logs,
        log_file=path,
    )
    echo(f"logging to file: {path}", err=True)
    return path


def _fail(message: str, *, code: int = _RUNTIME_ERROR_EXIT_CODE) -> Exit:
    """Emit ``message`` to stderr and return a typer :class:`Exit` of ``code``."""
    echo(message, err=True)
    return Exit(code=code)


def _validate_year_window(value: int | None) -> int | None:
    """Return ``value`` unchanged when it lies in ``[0, 100]``; raise otherwise.

    ``None`` (the flag was omitted) passes through unchanged so callers can
    fall back to the matching config's default.
    """
    if value is None:
        return None
    if value < _YEAR_WINDOW_MIN or value > _YEAR_WINDOW_MAX:
        raise BadParameter(
            f"--year-window: must be between {_YEAR_WINDOW_MIN} and "
            f"{_YEAR_WINDOW_MAX} (got {value})",
        )
    return value


def _load_default_matching_config() -> MatchingConfig:
    """Load the shipped ``matching.yaml`` defaults."""
    resource = files("pd_matcher.config.defaults") / "matching.yaml"
    with as_file(resource) as path:
        return load_matching_config(path)


def _load_default_pairing_config() -> PairingConfig:
    """Load the shipped ``field_pairings.yaml`` defaults."""
    resource = files("pd_matcher.config.defaults") / "field_pairings.yaml"
    with as_file(resource) as path:
        return load_pairing_config(path)


def _format_build_report(report: BuildReport, out_path: Path) -> str:
    """Render a :class:`BuildReport` as a small human-readable block."""
    skipped = "yes" if report.skipped else "no"
    return (
        f"Index: {out_path}\n"
        f"  skipped: {skipped}\n"
        f"  registrations: {report.registrations_written}\n"
        f"  renewals: {report.renewals_written}\n"
        f"  renewal joins: {report.renewal_joins}\n"
        f"  year buckets: {report.year_buckets}\n"
        f"  duration: {report.duration_seconds:.2f}s"
    )


def _format_index_stats(stats: IndexStats) -> str:
    """Render an :class:`IndexStats` as a small human-readable table."""
    return (
        f"  schema_version: {stats.schema_version}\n"
        f"  source_hash: {stats.source_hash}\n"
        f"  build_timestamp: {stats.build_timestamp}\n"
        f"  registrations: {stats.registrations_written}\n"
        f"  renewals: {stats.renewals_written}\n"
        f"  renewal joins: {stats.renewal_joins}\n"
        f"  year buckets: {stats.year_buckets}"
    )


def _format_run_report(report: RunReport, out_path: Path) -> str:
    """Render a :class:`RunReport` as a small human-readable summary."""
    return (
        f"Output: {out_path}\n"
        f"  records processed: {report.records_processed}\n"
        f"  records written: {report.records_written}\n"
        f"  records enqueued: {report.records_enqueued}\n"
        f"  duration: {report.duration_seconds:.2f}s\n"
        f"  interrupted: {'yes' if report.interrupted else 'no'}"
    )


def _format_prepare_report(report: PrepareReport) -> str:
    """Render a :class:`PrepareReport` as a small human-readable summary."""
    return (
        f"Prepared: {report.out_dir}\n"
        f"  skipped: {'yes' if report.skipped else 'no'}\n"
        f"  records: {report.total_records}\n"
        f"  chunks: {report.chunk_count}\n"
        f"  chunk size: {report.chunk_size}\n"
        f"  duration: {report.duration_seconds:.2f}s"
    )


def _format_sweep_row(point: ThresholdPoint) -> str:
    """Render one :class:`ThresholdPoint` as a compact table row."""
    return (
        f"    t={point.threshold:.2f}  "
        f"P={point.precision:.3f}  R={point.recall:.3f}  F1={point.f1:.3f}  "
        f"TP={point.true_positives}  FP={point.false_positives}  "
        f"FN={point.false_negatives}"
    )


def _format_sweep(sweep: tuple[ThresholdPoint, ...]) -> str:
    """Render a subset of the sweep grid (every Nth point) as a small table."""
    if not sweep:
        return "    (no scored pairs)"
    preview = sweep[::_SWEEP_PREVIEW_STEP]
    if preview[-1].threshold != sweep[-1].threshold:
        preview = (*preview, sweep[-1])
    return "\n".join(_format_sweep_row(point) for point in preview)


def _format_eval_report(report: EvalReport) -> str:
    """Render an :class:`EvalReport` as a small human-readable summary."""
    return (
        f"  pairs evaluated: {report.pairs_evaluated} "
        f"(positive={report.pairs_positive} negative={report.pairs_negative} "
        f"unsure_excluded={report.pairs_unsure_excluded})\n"
        f"  marcs evaluated: {report.marcs_evaluated} "
        f"(with top={report.marcs_with_matcher_top} "
        f"correct top={report.marcs_with_correct_top})\n"
        f"  precision: {report.precision:.3f}\n"
        f"  recall: {report.recall:.3f}\n"
        f"  f1: {report.f1:.3f}\n"
        f"  auc_roc: {report.auc_roc:.3f}\n"
        f"  average_precision: {report.average_precision:.3f}\n"
        f"  threshold sweep:\n"
        f"{_format_sweep(report.threshold_sweep)}\n"
        f"  elapsed: {report.elapsed_seconds:.2f}s"
    )


def _load_calibrator(parent: Path) -> PlattCalibrator | None:
    """Load a Platt calibrator from ``<parent>/calibrator.msgpack`` if present."""
    candidate = parent / _CALIBRATOR_NAME
    if not candidate.exists():
        return None
    return load_calibrator(candidate)


@index_app.command("build")
def index_build(
    reg_dir: Annotated[
        Path,
        Option("--reg-dir", help="Directory containing NYPL registration XML files."),
    ],
    ren_dir: Annotated[
        Path,
        Option("--ren-dir", help="Directory containing NYPL renewal TSV files."),
    ],
    out: Annotated[
        Path,
        Option("--out", help="LMDB environment directory to create or update."),
    ],
    force: Annotated[
        bool,
        Option("--force/--no-force", help="Rebuild even if the existing index is current."),
    ] = False,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Build the LMDB index from NYPL registration and renewal sources.

    Examples:
        pd-matcher index build \\
            --reg-dir data/nypl-reg/xml \\
            --ren-dir data/nypl-ren/data \\
            --out caches/cce.lmdb
    """
    _enable_log_file("index-build", log_file)
    if not reg_dir.is_dir():
        raise _fail(f"--reg-dir does not exist or is not a directory: {reg_dir}")
    if not ren_dir.is_dir():
        raise _fail(f"--ren-dir does not exist or is not a directory: {ren_dir}")
    try:
        report = build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out, force=force)
    except OSError as exc:
        raise _fail(f"index build failed: {exc}") from exc
    echo(_format_build_report(report, out))


@index_app.command("info")
def index_info(
    lmdb_path: Annotated[
        Path,
        Option("--lmdb-path", help="LMDB environment directory to inspect."),
    ],
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Print counts, build time, and source hashes for an existing index.

    Examples:
        pd-matcher index info --lmdb-path caches/cce.lmdb
    """
    _enable_log_file("index-info", log_file)
    if not lmdb_path.exists():
        raise _fail(f"--lmdb-path does not exist: {lmdb_path}")
    try:
        with NyplIndexLookup(lmdb_path) as lookup:
            stats = lookup.stats()
    except (OSError, RuntimeError) as exc:
        raise _fail(f"index info failed: {exc}") from exc
    echo(f"Index: {lmdb_path}")
    echo(_format_index_stats(stats))


@app.command("prepare-marc")
def prepare_marc_command(
    marc: Annotated[
        Path,
        Option("--marc", help="MARCXML file OR directory of *.xml shards."),
    ],
    out: Annotated[
        Path,
        Option("--out", help="Destination directory for pickled chunks + manifest."),
    ],
    chunk_size: Annotated[
        int,
        Option("--chunk-size", help="Target records per chunk."),
    ] = 1000,
    force: Annotated[
        bool,
        Option("--force/--no-force", help="Rebuild even when the prepared cache is current."),
    ] = False,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Stream MARCXML into re-runnable pickled chunks for `match --prepared`.

    Examples:
        pd-matcher prepare-marc \\
            --marc data/candidates/eng \\
            --out caches/prepared-eng
    """
    _enable_log_file("prepare-marc", log_file)
    if chunk_size < 1:
        raise _fail(
            f"--chunk-size: must be a positive integer (got {chunk_size})",
            code=_ARG_ERROR_EXIT_CODE,
        )
    if not marc.exists():
        raise _fail(f"--marc does not exist: {marc}")
    try:
        report = prepare_marc(marc, out, chunk_size=chunk_size, force=force)
    except OSError as exc:
        raise _fail(f"prepare-marc failed: {exc}") from exc
    echo(_format_prepare_report(report))


@app.command("match")
def match(
    index: Annotated[Path, Option("--index", help="LMDB index directory.")],
    out: Annotated[Path, Option("--out", help="Output CSV path.")],
    marc: Annotated[
        Path | None,
        Option("--marc", help="MARC XML file (mutually exclusive with --prepared)."),
    ] = None,
    prepared: Annotated[
        Path | None,
        Option(
            "--prepared",
            help="Prepared-chunk directory from `prepare-marc` (mutually exclusive with --marc).",
        ),
    ] = None,
    verbose: Annotated[
        int,
        Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase logging: -v per-worker heartbeats, -vv per-record hits.",
        ),
    ] = 0,
    workers: Annotated[
        int | None,
        Option("--workers", help="Number of worker processes (default: cpu_count - 1)."),
    ] = None,
    year_window: Annotated[
        int | None,
        Option("--year-window", help="Override the matching config's year_window."),
    ] = None,
    min_score: Annotated[
        float | None,
        Option("--min-score", help="Override the matching config's min_combined_score."),
    ] = None,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Match MARC records against the NYPL index and write a CSV linkage report.

    Examples:
        pd-matcher match \\
            --marc data/sample.marcxml \\
            --index caches/cce.lmdb \\
            --out /tmp/results.csv \\
            --workers 4
    """
    resolved_log_file = _enable_log_file("match", log_file)
    if (marc is None) == (prepared is None):
        raise _fail(
            "exactly one of --marc or --prepared is required",
            code=_ARG_ERROR_EXIT_CODE,
        )
    expected_total: int | None = None
    if marc is not None and not marc.is_file():
        raise _fail(f"--marc does not exist or is not a file: {marc}")
    if prepared is not None:
        if not prepared.is_dir():
            raise _fail(f"--prepared does not exist or is not a directory: {prepared}")
        try:
            expected_total = read_manifest(prepared).total_records
        except (OSError, DecodeError) as exc:
            raise _fail(f"--prepared has no readable manifest: {exc}") from exc
    if not index.exists():
        raise _fail(f"--index does not exist: {index}")
    try:
        matching_config = _load_default_matching_config()
    except ConfigError as exc:
        raise _fail(f"failed to load matching defaults: {exc}") from exc
    if year_window is not None or min_score is not None:
        matching_config = MatchingConfig(
            title_weight=matching_config.title_weight,
            author_weight=matching_config.author_weight,
            publisher_weight=matching_config.publisher_weight,
            year_weight=matching_config.year_weight,
            edition_weight=matching_config.edition_weight,
            lccn_weight=matching_config.lccn_weight,
            isbn_weight=matching_config.isbn_weight,
            extent_weight=matching_config.extent_weight,
            volume_weight=matching_config.volume_weight,
            year_window=year_window if year_window is not None else matching_config.year_window,
            min_combined_score=(
                min_score if min_score is not None else matching_config.min_combined_score
            ),
            scorer=matching_config.scorer,
        )
    try:
        pairing_config = _load_default_pairing_config()
    except ConfigError as exc:
        raise _fail(f"failed to load pairing defaults: {exc}") from exc
    idf_cache_path = index.parent / _IDF_CACHE_NAME
    try:
        idf = load_or_build_idf(idf_cache_path, lambda: NyplIndexLookup(index))
    except OSError as exc:
        raise _fail(f"failed to load/build IDF table: {exc}") from exc
    calibrator = _load_calibrator(index.parent)
    try:
        report = run_match(
            marc_path=marc,
            prepared_dir=prepared,
            expected_total=expected_total,
            index_path=index,
            output_path=out,
            matching_config=matching_config,
            pairing_config=pairing_config,
            idf=idf,
            calibrator=calibrator,
            workers=workers,
            report_interval_seconds=5.0,
            verbosity=verbose,
            log_level=_LOG_SETTINGS.level,
            json_logs=_LOG_SETTINGS.json_logs,
            log_file=resolved_log_file,
        )
    except OSError as exc:
        raise _fail(f"match run failed: {exc}") from exc
    if report.interrupted:
        echo(f"interrupted, partial output at {out}", err=True)
        raise Exit(code=_INTERRUPTED_EXIT_CODE)
    echo(_format_run_report(report, out))


@app.command("eval")
def eval_(
    index: Annotated[Path, Option("--index", help="LMDB index directory.")],
    vault: Annotated[
        Path,
        Option("--vault", help="Label vault JSONL path."),
    ] = _DEFAULT_VAULT_PATH,
    pool: Annotated[
        Path,
        Option("--pool", help="MARC candidate pool root directory (<pool>/<lang>/*.xml)."),
    ] = _DEFAULT_POOL_PATH,
    report: Annotated[
        Path | None,
        Option("--report", help="Optional path for a JSON copy of the eval report."),
    ] = None,
    year_window: Annotated[
        int | None,
        Option(
            "--year-window",
            help="Override the matching config's year_window for this eval run.",
        ),
    ] = None,
    log_file: Annotated[
        Path | None,
        Option("--log-file", help="Override the auto-generated log file path."),
    ] = None,
) -> None:
    """Evaluate the matcher's linkage against the label vault.

    Examples:
        pd-matcher eval \\
            --vault data/label_vault.jsonl \\
            --pool data/candidates \\
            --index caches/cce.lmdb \\
            --report /tmp/eval.json
    """
    _enable_log_file("eval", log_file)
    year_window = _validate_year_window(year_window)
    if not vault.is_file():
        raise _fail(f"--vault does not exist or is not a file: {vault}")
    if not pool.is_dir():
        raise _fail(f"--pool does not exist or is not a directory: {pool}")
    if not index.exists():
        raise _fail(f"--index does not exist: {index}")
    try:
        matching_config = _load_default_matching_config()
    except ConfigError as exc:
        raise _fail(f"failed to load matching defaults: {exc}") from exc
    if year_window is not None:
        matching_config = MatchingConfig(
            title_weight=matching_config.title_weight,
            author_weight=matching_config.author_weight,
            publisher_weight=matching_config.publisher_weight,
            year_weight=matching_config.year_weight,
            edition_weight=matching_config.edition_weight,
            lccn_weight=matching_config.lccn_weight,
            isbn_weight=matching_config.isbn_weight,
            extent_weight=matching_config.extent_weight,
            volume_weight=matching_config.volume_weight,
            year_window=year_window,
            min_combined_score=matching_config.min_combined_score,
            scorer=matching_config.scorer,
        )
    try:
        pairing_config = _load_default_pairing_config()
    except ConfigError as exc:
        raise _fail(f"failed to load pairing defaults: {exc}") from exc
    try:
        eval_report = run_eval(
            vault_path=vault,
            pool_path=pool,
            index_path=index,
            matching_config=matching_config,
            pairing_config=pairing_config,
        )
    except OSError as exc:
        raise _fail(f"eval run failed: {exc}") from exc
    echo("Eval report:")
    echo(_format_eval_report(eval_report))
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(dumps(to_builtins(eval_report), indent=2), encoding="utf-8")


@app.command("train-scorer")
def train_scorer() -> None:
    """Train the Phase 9 learned scorer (placeholder).

    The learned scorer is implemented in Phase 9; the current branch only
    provides the CLI skeleton.
    """
    raise _fail(
        "train-scorer is implemented in Phase 9; current branch only provides the CLI skeleton",
        code=_ARG_ERROR_EXIT_CODE,
    )


__all__ = ["app", "index_app"]
