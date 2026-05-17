"""Typer entry point for the ``pd-matcher`` command.

Phase 1 wires up argument parsing and logging only; every subcommand exits
with code 2 and a "not yet implemented" message. Subsequent phases will
replace those bodies with real logic.
"""

from pathlib import Path
from typing import Annotated

from typer import Exit
from typer import Option
from typer import Typer
from typer import echo

_NOT_IMPLEMENTED_EXIT_CODE: int = 2


app: Typer = Typer(help="MARC ↔ NYPL public-domain matcher", no_args_is_help=True)
index_app: Typer = Typer(help="Build and inspect the LMDB NYPL index.", no_args_is_help=True)
app.add_typer(index_app, name="index")


@app.callback()
def _main(
    log_level: Annotated[
        str,
        Option("--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR)."),
    ] = "INFO",
    json_logs: Annotated[
        bool,
        Option("--json-logs/--no-json-logs", help="Emit logs as JSON (one object per line)."),
    ] = False,
) -> None:
    """Initialize logging before any subcommand runs."""
    from pd_matcher.logging_config import configure_logging

    configure_logging(level=log_level, json_output=json_logs)


def _not_implemented(command: str) -> None:
    """Print a uniform "not yet implemented" message and exit with code 2."""
    echo(f"{command}: not yet implemented (Phase 1 skeleton)", err=True)
    raise Exit(code=_NOT_IMPLEMENTED_EXIT_CODE)


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
) -> None:
    """Build the LMDB index from NYPL registration and renewal sources."""
    del reg_dir, ren_dir, out
    _not_implemented("index build")


@index_app.command("info")
def index_info(
    lmdb_path: Annotated[
        Path,
        Option("--lmdb-path", help="LMDB environment directory to inspect."),
    ],
) -> None:
    """Print counts, build time, and source hashes for an existing index."""
    del lmdb_path
    _not_implemented("index info")


@app.command("match")
def match(
    marc: Annotated[Path, Option("--marc", help="MARC XML file or directory.")],
    index: Annotated[Path, Option("--index", help="LMDB index directory.")],
    out: Annotated[Path, Option("--out", help="Output CSV path.")],
    workers: Annotated[int, Option("--workers", help="Number of worker processes.")] = 1,
    year_window: Annotated[
        int,
        Option("--year-window", help="Year-bucket window (± years) used for blocking."),
    ] = 2,
    min_score: Annotated[
        float,
        Option("--min-score", help="Minimum combined score to record a match."),
    ] = 70.0,
) -> None:
    """Match MARC records against the NYPL index and write a CSV report."""
    del marc, index, out, workers, year_window, min_score
    _not_implemented("match")


@app.command("eval")
def eval_(
    ground_truth: Annotated[
        Path,
        Option("--ground-truth", help="Ground-truth CSV (combined_ground_truth.csv)."),
    ],
    index: Annotated[Path, Option("--index", help="LMDB index directory.")],
    report: Annotated[
        Path | None,
        Option("--report", help="Optional path for the JSON eval report."),
    ] = None,
) -> None:
    """Evaluate the matcher against the ground-truth pairs."""
    del ground_truth, index, report
    _not_implemented("eval")


__all__ = ["app", "index_app"]
