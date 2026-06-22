"""Free-disk-space guard for the streaming dump downloads.

A multi-dump run (``build-corpus`` streams the whole catalog) writes one
compressed archive at a time to a temp directory and appends survivors to an
output file. Neither lands its full size in memory, but both consume disk, and
a long run can silently exhaust the filesystem. :class:`DiskSpaceGuard` makes
that failure loud and safe: it asserts, before each download and periodically
mid-download, that every relevant filesystem still has at least a configured
number of free bytes, raising :class:`InsufficientDiskSpaceError` (which names
the path, the actual free space, and the threshold) the moment headroom runs
out so the caller can clean up and finalize a valid partial corpus.

A guard built with ``min_free_bytes <= 0`` is disabled: its assertions are
no-ops, which is how ``--min-free-space-mb 0`` turns the feature off entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import disk_usage

_BYTES_PER_MB = 1 << 20
_BYTES_PER_GB = 1 << 30


class InsufficientDiskSpaceError(RuntimeError):
    """Raised when a path's filesystem has less free space than required.

    Producers that maintain progress counts (e.g. ``build-corpus``) may stamp
    :attr:`records_written` / :attr:`dumps_written` before re-raising so the
    caller can report how much valid output survived.
    """

    records_written: int | None = None
    dumps_written: int | None = None


@dataclass(frozen=True, slots=True)
class DiskSpaceGuard:
    """Assert that filesystems retain a minimum number of free bytes.

    Attributes:
        min_free_bytes: The required free-space floor per filesystem. A value
            of zero or less disables the guard, making :meth:`ensure` a no-op.
    """

    min_free_bytes: int

    @staticmethod
    def from_megabytes(megabytes: int) -> DiskSpaceGuard:
        """Build a guard from a threshold expressed in whole megabytes.

        Args:
            megabytes: Required free space per filesystem in MB; ``0`` (or
                negative) yields a disabled guard.

        Returns:
            A :class:`DiskSpaceGuard` with the equivalent byte threshold.
        """
        return DiskSpaceGuard(min_free_bytes=megabytes * _BYTES_PER_MB)

    @property
    def enabled(self) -> bool:
        """Whether the guard performs any checks (``min_free_bytes > 0``)."""
        return self.min_free_bytes > 0

    def free_bytes(self, path: Path) -> int:
        """Return the free bytes on the filesystem backing ``path``.

        The nearest existing ancestor of ``path`` is queried, so a not-yet-created
        temp file or output directory still reports the right filesystem.

        Args:
            path: A filesystem location (need not exist yet).

        Returns:
            The number of free bytes on that location's filesystem.
        """
        return disk_usage(_existing_ancestor(path)).free

    def ensure(self, *paths: Path) -> None:
        """Assert every path's filesystem has at least ``min_free_bytes`` free.

        Args:
            *paths: Filesystem locations to check (each resolved to its nearest
                existing ancestor before querying). Duplicate filesystems are
                still checked once per argument; callers pass only the few paths
                that matter, so the redundant ``statvfs`` is negligible.

        Raises:
            InsufficientDiskSpaceError: If any path's filesystem has fewer than
                ``min_free_bytes`` free, naming the path, the free space, and
                the threshold.
        """
        if not self.enabled:
            return
        for path in paths:
            free = self.free_bytes(path)
            if free < self.min_free_bytes:
                raise InsufficientDiskSpaceError(
                    f"insufficient disk space at {path}: "
                    f"{human_bytes(free)} free, "
                    f"need at least {human_bytes(self.min_free_bytes)}"
                )


def _existing_ancestor(path: Path) -> Path:
    """Return ``path`` or its nearest ancestor that exists on disk.

    ``shutil.disk_usage`` requires an existing path; a temp file or an output
    directory may not exist yet, so the chain is walked upward until an existing
    directory (ultimately the root) is found.
    """
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return candidate
        candidate = parent
    return candidate


def human_bytes(num_bytes: int) -> str:
    """Render a byte count as a human-readable ``MB``/``GB`` string."""
    if num_bytes >= _BYTES_PER_GB:
        return f"{num_bytes / _BYTES_PER_GB:.2f} GB"
    return f"{num_bytes / _BYTES_PER_MB:.2f} MB"
