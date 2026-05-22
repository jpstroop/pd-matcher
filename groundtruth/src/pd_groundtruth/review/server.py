"""Launch the review UI with uvicorn over a bound review database.

Thin glue between the Typer ``review`` subcommand and uvicorn: it binds the
database path into a freshly created app and serves it on the requested
host/port. Kept out of the unit-test path (and default coverage) because it
does nothing but call :func:`uvicorn.run`.
"""

from pathlib import Path

from uvicorn import run as uvicorn_run

from pd_groundtruth.review.app import create_app


def serve(db_path: Path, host: str, port: int) -> None:
    """Bind ``db_path`` into the app and serve it on ``host:port``.

    A ``Ctrl-C`` (``KeyboardInterrupt``) returns cleanly after uvicorn's own
    graceful shutdown completes, suppressing only the cosmetic interrupt
    traceback. Any other failure from :func:`uvicorn.run` (e.g. a bind error)
    propagates so genuine problems stay visible.

    Args:
        db_path: Path to the SQLite ``review.db`` to label against.
        host: Interface to bind (default a loopback address for local use).
        port: TCP port to listen on.

    Raises:
        FileNotFoundError: If ``db_path`` does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"review database not found: {db_path}")
    application = create_app(db_path)
    try:
        uvicorn_run(application, host=host, port=port)
    except KeyboardInterrupt:
        print("Stopped.")


__all__ = [
    "serve",
]
