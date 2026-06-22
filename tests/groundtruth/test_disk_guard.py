"""Unit tests for the free-disk-space guard (no real disk dependency)."""

from collections import namedtuple
from pathlib import Path

from pytest import MonkeyPatch
from pytest import raises

from pd_groundtruth.disk_guard import DiskSpaceGuard
from pd_groundtruth.disk_guard import InsufficientDiskSpaceError
from pd_groundtruth.disk_guard import human_bytes

_GB = 1 << 30
_MB = 1 << 20

_Usage = namedtuple("_Usage", ("total", "used", "free"))


def _fake_usage(free: int) -> _Usage:
    """Return a ``shutil.disk_usage``-shaped triple with the given free bytes."""
    return _Usage(total=free * 2, used=free, free=free)


def _patch_free(monkeypatch: MonkeyPatch, free_by_path: dict[Path, int]) -> None:
    """Patch ``shutil.disk_usage`` (as imported by disk_guard) per path."""

    def _usage(path: str | Path) -> _Usage:
        return _fake_usage(free_by_path[Path(path)])

    monkeypatch.setattr("pd_groundtruth.disk_guard.disk_usage", _usage)


def test_from_megabytes_converts_to_bytes() -> None:
    guard = DiskSpaceGuard.from_megabytes(2048)
    assert guard.min_free_bytes == 2048 * _MB
    assert guard.enabled is True


def test_zero_megabytes_disables_guard() -> None:
    guard = DiskSpaceGuard.from_megabytes(0)
    assert guard.enabled is False
    assert guard.min_free_bytes == 0


def test_negative_megabytes_disables_guard() -> None:
    guard = DiskSpaceGuard.from_megabytes(-5)
    assert guard.enabled is False


def test_ensure_passes_when_above_threshold(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _patch_free(monkeypatch, {tmp_path: 3 * _GB})
    DiskSpaceGuard.from_megabytes(2048).ensure(tmp_path)


def test_ensure_raises_below_threshold_naming_path_free_and_floor(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _patch_free(monkeypatch, {tmp_path: 100 * _MB})
    with raises(InsufficientDiskSpaceError) as excinfo:
        DiskSpaceGuard.from_megabytes(2048).ensure(tmp_path)
    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "100.00 MB free" in message
    assert "2.00 GB" in message


def test_ensure_checks_every_path_and_raises_on_the_low_one(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    temp_dir = tmp_path / "temp"
    out_dir = tmp_path / "out"
    temp_dir.mkdir()
    out_dir.mkdir()
    _patch_free(monkeypatch, {temp_dir: 5 * _GB, out_dir: 10 * _MB})
    with raises(InsufficientDiskSpaceError, match=str(out_dir)):
        DiskSpaceGuard.from_megabytes(2048).ensure(temp_dir, out_dir)


def test_disabled_guard_never_queries_disk(monkeypatch: MonkeyPatch) -> None:
    def _boom(_path: str | Path) -> object:
        raise AssertionError("disk_usage must not be called when disabled")

    monkeypatch.setattr("pd_groundtruth.disk_guard.disk_usage", _boom)
    DiskSpaceGuard.from_megabytes(0).ensure(Path("/nonexistent"))


def test_free_bytes_walks_up_to_existing_ancestor(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    missing = tmp_path / "does" / "not" / "exist" / "file.tar.gz"
    seen: list[Path] = []

    def _usage(path: str | Path) -> _Usage:
        seen.append(Path(path))
        return _fake_usage(7 * _GB)

    monkeypatch.setattr("pd_groundtruth.disk_guard.disk_usage", _usage)
    assert DiskSpaceGuard.from_megabytes(1).free_bytes(missing) == 7 * _GB
    assert seen == [tmp_path]


def test_existing_ancestor_falls_back_to_root(monkeypatch: MonkeyPatch) -> None:
    seen: list[Path] = []

    def _exists(_self: Path) -> bool:
        return False

    def _usage(path: str | Path) -> _Usage:
        seen.append(Path(path))
        return _fake_usage(_GB)

    monkeypatch.setattr(Path, "exists", _exists)
    monkeypatch.setattr("pd_groundtruth.disk_guard.disk_usage", _usage)
    DiskSpaceGuard.from_megabytes(1).free_bytes(Path("/a/b/c"))
    assert seen == [Path("/")]


def test_human_bytes_renders_mb_and_gb() -> None:
    assert human_bytes(512 * _MB) == "512.00 MB"
    assert human_bytes(2 * _GB) == "2.00 GB"
    assert human_bytes(_GB) == "1.00 GB"
