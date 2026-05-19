"""Typer entry point for the ``pd-matcher`` command.

Phase 7 wires every subcommand to its implementation module. Errors are
reported on stderr with these exit codes:

* ``0`` — success
* ``1`` — runtime error (missing file, bad config, IO failure)
* ``2`` — argument error (typer-default for parse failures, plus the
  ``--as-of`` year validator and the not-yet-implemented stubs)
* ``130`` — interrupted by SIGINT
"""

from datetime import date
from importlib.resources import as_file
from importlib.resources import files
from json import dumps
from logging import WARNING
from pathlib import Path
from typing import Annotated

from msgspec import to_builtins
from typer import BadParameter
from typer import Exit
from typer import Option
from typer import Typer
from typer import echo

from pd_matcher.config.loader import ConfigError
from pd_matcher.config.loader import load_copyright_rules
from pd_matcher.config.loader import load_matching_config
from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.eval.ground_truth import EvalReport
from pd_matcher.eval.ground_truth import run_eval
from pd_matcher.index.builder import BuildReport
from pd_matcher.index.builder import build_index
from pd_matcher.index.lookup import IndexStats
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.logging_config import configure_logging
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.calibrator import load_calibrator
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.workers import RunReport
from pd_matcher.workers import run_match

_ARG_ERROR_EXIT_CODE: int = 2
_RUNTIME_ERROR_EXIT_CODE: int = 1
_INTERRUPTED_EXIT_CODE: int = 130

_IDF_CACHE_NAME: str = "idf.msgpack"
_CALIBRATOR_NAME: str = "calibrator.msgpack"

_AS_OF_MIN_YEAR: int = 1923
_AS_OF_MAX_YEAR: int = 2100


app: Typer = Typer(
    help="MARC ↔ NYPL public-domain matcher.",
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
    if quiet:
        from logging import getLogger

        getLogger().setLevel(WARNING)


def _fail(message: str, *, code: int = _RUNTIME_ERROR_EXIT_CODE) -> Exit:
    """Emit ``message`` to stderr and return a typer :class:`Exit` of ``code``."""
    echo(message, err=True)
    return Exit(code=code)


def _parse_as_of(value: str | None) -> int:
    """Parse an ``--as-of`` flag value or default to the current year.

    The data this CLI consumes (CCE registrations, MARC publication info,
    renewal years) is year-granular, so the flag accepts a four-digit
    year. ``1923`` is the lower bound of the CCE corpus; ``2100`` is a
    typo-catcher upper bound. Values outside the range, or non-integer
    strings, raise :class:`typer.BadParameter` with a clear message.
    """
    if value is None:
        return date.today().year
    try:
        year = int(value, 10)
    except ValueError as exc:
        raise BadParameter(
            f"--as-of: expected a four-digit year YYYY (got {value!r})",
        ) from exc
    if year < _AS_OF_MIN_YEAR or year > _AS_OF_MAX_YEAR:
        raise BadParameter(
            f"--as-of: year must be between {_AS_OF_MIN_YEAR} and {_AS_OF_MAX_YEAR} (got {year})",
        )
    return year


def _load_default_matching_config() -> MatchingConfig:
    """Load the shipped ``matching.yaml`` defaults."""
    resource = files("pd_matcher.config.defaults") / "matching.yaml"
    with as_file(resource) as path:
        return load_matching_config(path)


def _load_default_ruleset_path() -> Path:
    """Materialize a real filesystem path for the shipped copyright rules.

    ``as_file`` keeps the temporary copy alive until the context exits;
    the callable returns the path eagerly because the loader reads the
    file immediately.
    """
    resource = files("pd_matcher.config.defaults") / "copyright_rules.yaml"
    with as_file(resource) as path:
        return Path(path)


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
        f"  interrupted: {'yes' if report.interrupted else 'no'}\n"
        f"  by status: {report.by_status}"
    )


def _format_eval_report(report: EvalReport) -> str:
    """Render an :class:`EvalReport` as a small human-readable summary."""
    return (
        f"  rows evaluated: {report.rows_evaluated}\n"
        f"  predicted match: {report.rows_with_predicted_match}\n"
        f"  ground-truth match: {report.rows_with_ground_truth_match}\n"
        f"  agreeing: {report.rows_agreeing}\n"
        f"  precision: {report.precision:.3f}\n"
        f"  recall: {report.recall:.3f}\n"
        f"  f1: {report.f1:.3f}\n"
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
) -> None:
    """Build the LMDB index from NYPL registration and renewal sources.

    Examples:
        pd-matcher index build \\
            --reg-dir data/nypl-reg/xml \\
            --ren-dir data/nypl-ren/data \\
            --out caches/nypl.lmdb
    """
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
) -> None:
    """Print counts, build time, and source hashes for an existing index.

    Examples:
        pd-matcher index info --lmdb-path caches/nypl.lmdb
    """
    if not lmdb_path.exists():
        raise _fail(f"--lmdb-path does not exist: {lmdb_path}")
    try:
        with NyplIndexLookup(lmdb_path) as lookup:
            stats = lookup.stats()
    except (OSError, RuntimeError) as exc:
        raise _fail(f"index info failed: {exc}") from exc
    echo(f"Index: {lmdb_path}")
    echo(_format_index_stats(stats))


@app.command("match")
def match(
    marc: Annotated[Path, Option("--marc", help="MARC XML file.")],
    index: Annotated[Path, Option("--index", help="LMDB index directory.")],
    out: Annotated[Path, Option("--out", help="Output CSV path.")],
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
    as_of: Annotated[
        str | None,
        Option(
            "--as-of",
            help=(
                "Reference year (YYYY) for the moving-wall calculation."
                " Defaults to the current year."
            ),
        ),
    ] = None,
) -> None:
    """Match MARC records against the NYPL index and write a CSV report.

    Examples:
        pd-matcher match \\
            --marc data/sample.marcxml \\
            --index caches/nypl.lmdb \\
            --out /tmp/results.csv \\
            --workers 4
    """
    as_of_year = _parse_as_of(as_of)
    if not marc.is_file():
        raise _fail(f"--marc does not exist or is not a file: {marc}")
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
            year_window=year_window if year_window is not None else matching_config.year_window,
            min_combined_score=(
                min_score if min_score is not None else matching_config.min_combined_score
            ),
            scorer=matching_config.scorer,
        )
    try:
        ruleset_path = _load_default_ruleset_path()
        ruleset = load_copyright_rules(ruleset_path)
    except ConfigError as exc:
        raise _fail(f"failed to load copyright rules: {exc}") from exc
    copyright_config = CopyrightAssessmentConfig(as_of_year=as_of_year)
    idf_cache_path = index.parent / _IDF_CACHE_NAME
    try:
        idf = load_or_build_idf(idf_cache_path, lambda: NyplIndexLookup(index))
    except OSError as exc:
        raise _fail(f"failed to load/build IDF table: {exc}") from exc
    calibrator = _load_calibrator(index.parent)
    try:
        report = run_match(
            marc_path=marc,
            index_path=index,
            output_path=out,
            matching_config=matching_config,
            copyright_config=copyright_config,
            ruleset=ruleset,
            idf=idf,
            calibrator=calibrator,
            workers=workers,
            report_interval_seconds=5.0,
        )
    except OSError as exc:
        raise _fail(f"match run failed: {exc}") from exc
    if report.interrupted:
        echo(f"interrupted, partial output at {out}", err=True)
        raise Exit(code=_INTERRUPTED_EXIT_CODE)
    echo(_format_run_report(report, out))


@app.command("eval")
def eval_(
    ground_truth: Annotated[
        Path,
        Option("--ground-truth", help="Ground-truth CSV (combined_ground_truth.csv shape)."),
    ],
    index: Annotated[Path, Option("--index", help="LMDB index directory.")],
    report: Annotated[
        Path | None,
        Option("--report", help="Optional path for a JSON copy of the eval report."),
    ] = None,
    as_of: Annotated[
        str | None,
        Option(
            "--as-of",
            help=(
                "Reference year (YYYY) for the moving-wall calculation."
                " Defaults to the current year."
            ),
        ),
    ] = None,
    limit: Annotated[
        int | None,
        Option("--limit", help="Evaluate at most this many rows."),
    ] = None,
) -> None:
    """Evaluate the matcher against the ground-truth pairs.

    Examples:
        pd-matcher eval \\
            --ground-truth data/combined_ground_truth.csv \\
            --index caches/nypl.lmdb \\
            --report /tmp/eval.json
    """
    as_of_year = _parse_as_of(as_of)
    if not ground_truth.is_file():
        raise _fail(f"--ground-truth does not exist or is not a file: {ground_truth}")
    if not index.exists():
        raise _fail(f"--index does not exist: {index}")
    try:
        matching_config = _load_default_matching_config()
    except ConfigError as exc:
        raise _fail(f"failed to load matching defaults: {exc}") from exc
    copyright_config = CopyrightAssessmentConfig(as_of_year=as_of_year)
    try:
        eval_report = run_eval(
            ground_truth_path=ground_truth,
            index_path=index,
            as_of_year=as_of_year,
            matching_config=matching_config,
            copyright_config=copyright_config,
            limit=limit,
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
