"""Unit tests for the matcher-build identifier helper."""

from importlib.metadata import PackageNotFoundError

from pytest import MonkeyPatch

from pd_groundtruth import matcher_version as module
from pd_groundtruth.matcher_version import matcher_version


def test_returns_short_hash_for_clean_checkout(monkeypatch: MonkeyPatch) -> None:
    def fake_git(*args: str) -> str | None:
        if args[0] == "rev-parse":
            return "abc1234"
        return ""

    monkeypatch.setattr(module, "_git", fake_git)
    assert matcher_version() == "abc1234"


def test_appends_dirty_suffix_when_tree_is_dirty(monkeypatch: MonkeyPatch) -> None:
    def fake_git(*args: str) -> str | None:
        if args[0] == "rev-parse":
            return "abc1234"
        return " M src/pd_groundtruth/enrich_vault.py"

    monkeypatch.setattr(module, "_git", fake_git)
    assert matcher_version() == "abc1234-dirty"


def test_falls_back_to_package_version_outside_git(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_git", lambda *_args: None)
    monkeypatch.setattr(module, "package_version", lambda _name: "9.9.9")
    assert matcher_version() == "9.9.9"


def test_falls_back_to_unknown_when_distribution_absent(monkeypatch: MonkeyPatch) -> None:
    def raise_not_found(_name: str) -> str:
        raise PackageNotFoundError(_name)

    monkeypatch.setattr(module, "_git", lambda *_args: None)
    monkeypatch.setattr(module, "package_version", raise_not_found)
    assert matcher_version() == "unknown"


def test_git_returns_none_on_failure(monkeypatch: MonkeyPatch) -> None:
    """The ``_git`` probe swallows a non-zero git invocation into ``None``."""

    def explode(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(module, "run", explode)
    assert module._git("rev-parse", "--short", "HEAD") is None


def test_git_returns_stripped_stdout(monkeypatch: MonkeyPatch) -> None:
    """The ``_git`` probe strips whitespace from a successful invocation."""

    class _Completed:
        stdout = "  abc1234\n"

    monkeypatch.setattr(module, "run", lambda *_a, **_k: _Completed())
    assert module._git("rev-parse", "--short", "HEAD") == "abc1234"


def test_git_real_invocation_does_not_raise() -> None:
    """Exercise the REAL subprocess call (no mock) to guard the ``run`` kwargs.

    A malformed invocation — e.g. ``capture_output=True`` together with an
    explicit ``stderr`` — raises ``ValueError`` from ``subprocess.run``, which is
    NOT one of the exceptions ``_git`` swallows, so it surfaces here. Mocking
    ``run`` (as the other tests do) hides that class of bug entirely. Returns the
    short hash inside a git checkout or ``None`` outside one; either is fine, but
    it must not raise.
    """
    result = module._git("rev-parse", "--short", "HEAD")
    assert result is None or (result != "" and result.strip() == result)
