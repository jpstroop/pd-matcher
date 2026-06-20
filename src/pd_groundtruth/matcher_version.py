"""Resolve a human-traceable identifier for the running matcher build.

``enrich-vault`` stamps every score it writes with the matcher build that
produced it, so a published linkage row can be traced back to the exact code
state. In a git checkout that identifier is the short commit hash, suffixed
with ``-dirty`` when the working tree has uncommitted changes (so scores
computed against unstaged edits are never mistaken for a clean build). Outside
a git checkout (an installed wheel, a sdist) the fallback is the package
version from the installed distribution metadata.

The git probe shells out: this is a publish/dev-time tool, not a hot path, so
the subprocess cost is irrelevant and avoids a libgit2 dependency.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from subprocess import DEVNULL
from subprocess import PIPE
from subprocess import CalledProcessError
from subprocess import run

_PACKAGE_NAME: str = "pd-matcher"
_UNKNOWN: str = "unknown"
_DIRTY_SUFFIX: str = "-dirty"


def _git(*args: str) -> str | None:
    """Run ``git`` with ``args`` and return stripped stdout, or ``None`` on failure.

    Returns ``None`` when git is not installed, the directory is not a git
    checkout, or the command exits non-zero — every "not in a checkout" path
    funnels here so the caller can fall back uniformly.
    """
    try:
        completed = run(
            ("git", *args),
            stdout=PIPE,
            stderr=DEVNULL,
            text=True,
            check=True,
        )
    except OSError, CalledProcessError:
        return None
    return completed.stdout.strip()


def _package_version() -> str:
    """Return the installed distribution version, or ``"unknown"`` if absent."""
    try:
        return package_version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _UNKNOWN


def matcher_version() -> str:
    """Return the matcher build identifier for stamping enriched scores.

    In a git checkout: the short commit hash, plus a ``-dirty`` suffix when the
    working tree has any staged or unstaged changes. Outside a checkout: the
    installed package version (or ``"unknown"`` when the distribution metadata
    is unavailable).
    """
    short_hash = _git("rev-parse", "--short", "HEAD")
    if short_hash is None:
        return _package_version()
    status = _git("status", "--porcelain")
    if status:
        return f"{short_hash}{_DIRTY_SUFFIX}"
    return short_hash


__all__ = [
    "matcher_version",
]
