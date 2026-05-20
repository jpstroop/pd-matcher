"""Typer entry point for the ground-truth acquisition tool."""

from logging import INFO
from logging import basicConfig
from pathlib import Path
from typing import Annotated

from typer import Option
from typer import Typer
from typer import echo

from pd_groundtruth.acquire import acquire
from pd_groundtruth.manifest import DEFAULT_MANIFEST_URL

app = Typer(add_completion=False, help="Acquire Princeton MARC ground-truth candidates.")


@app.callback()
def main() -> None:
    """Princeton MARC ground-truth acquisition CLI."""


@app.command(name="acquire")
def acquire_command(
    out_dir: Annotated[Path, Option("--out-dir", help="Directory for MARCXML shards.")],
    manifest_url: Annotated[
        str, Option("--manifest-url", help="Dump manifest JSON URL.")
    ] = DEFAULT_MANIFEST_URL,
    max_records: Annotated[
        int, Option("--max-records", help="Stop after this many eligible records.")
    ] = 50000,
    max_dumps: Annotated[
        int | None, Option("--max-dumps", help="Cap the number of dumps processed.")
    ] = None,
) -> None:
    """Stream dumps and write eligible records as MARCXML shards."""
    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    report = acquire(
        out_dir=out_dir,
        manifest_url=manifest_url,
        max_records=max_records,
        max_dumps=max_dumps,
    )
    echo(
        f"dumps_processed={report.dumps_processed} "
        f"records_scanned={report.records_scanned} "
        f"records_kept={report.records_kept} "
        f"shards_written={report.shards_written} "
        f"stopped_reason={report.stopped_reason}"
    )
