"""Tests for :mod:`pd_matcher.workers.thread_limits`."""

from os import environ

from pytest import MonkeyPatch

from pd_matcher.workers.thread_limits import limit_worker_threads
from pd_matcher.workers.thread_limits import numeric_thread_env
from pd_matcher.workers.thread_limits import pin_numeric_threads_in_env

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def test_numeric_thread_env_pins_every_var_to_one() -> None:
    assert numeric_thread_env() == dict.fromkeys(_THREAD_VARS, "1")


def test_pin_numeric_threads_in_env_exports_single_thread(
    monkeypatch: MonkeyPatch,
) -> None:
    for name in _THREAD_VARS:
        monkeypatch.delenv(name, raising=False)
    pin_numeric_threads_in_env()
    for name in _THREAD_VARS:
        assert environ[name] == "1"


def test_pin_numeric_threads_in_env_overrides_existing(
    monkeypatch: MonkeyPatch,
) -> None:
    for name in _THREAD_VARS:
        monkeypatch.setenv(name, "16")
    pin_numeric_threads_in_env()
    for name in _THREAD_VARS:
        assert environ[name] == "1"


def test_limit_worker_threads_caps_pool_to_one() -> None:
    from threadpoolctl import threadpool_info

    with limit_worker_threads():
        for pool in threadpool_info():
            assert pool["num_threads"] == 1
