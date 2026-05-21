"""Typer entry point for the ground-truth acquisition tool."""

from logging import INFO
from logging import basicConfig
from pathlib import Path
from typing import Annotated

from typer import Option
from typer import Typer
from typer import echo

from pd_groundtruth.acquire import acquire
from pd_groundtruth.acquire import default_min_year
from pd_groundtruth.manifest import DEFAULT_MANIFEST_URL

app = Typer(add_completion=False, help="Acquire Princeton MARC ground-truth candidates.")

_DEFAULT_PER_DECADE_CAP = 20000


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
