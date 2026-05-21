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

_DEFAULT_CAP_ENG = 40000
_DEFAULT_CAP_OTHER = 2500


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
    cap_eng: Annotated[
        int, Option("--cap-eng", help="Maximum English (eng) records to keep.")
    ] = _DEFAULT_CAP_ENG,
    cap_fre: Annotated[
        int, Option("--cap-fre", help="Maximum French (fre) records to keep.")
    ] = _DEFAULT_CAP_OTHER,
    cap_ger: Annotated[
        int, Option("--cap-ger", help="Maximum German (ger) records to keep.")
    ] = _DEFAULT_CAP_OTHER,
    cap_spa: Annotated[
        int, Option("--cap-spa", help="Maximum Spanish (spa) records to keep.")
    ] = _DEFAULT_CAP_OTHER,
    cap_ita: Annotated[
        int, Option("--cap-ita", help="Maximum Italian (ita) records to keep.")
    ] = _DEFAULT_CAP_OTHER,
    max_dumps: Annotated[
        int | None, Option("--max-dumps", help="Cap the number of dumps processed.")
    ] = None,
) -> None:
    """Stream dumps and write eligible records as per-language MARCXML shards."""
    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    caps = {
        "eng": cap_eng,
        "fre": cap_fre,
        "ger": cap_ger,
        "spa": cap_spa,
        "ita": cap_ita,
    }
    report = acquire(
        out_dir=out_dir,
        caps=caps,
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
