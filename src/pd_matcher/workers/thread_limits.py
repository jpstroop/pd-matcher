"""Pin numeric-library thread pools so the match worker pool stays CPU-bound.

The match command fans work across one ``spawn`` worker process per core. Each
worker re-imports LightGBM (bundled OpenMP) and numpy/BLAS, every one of which
defaults its internal thread pool to *all* cores. The product is catastrophic:
``workers x cores`` threads spin-waiting on ``cores`` CPUs, so the pool spends
its time on context switches instead of matching (~6 rec/min observed, with the
run dominated by kernel scheduling rather than scoring).

The fix pins each numeric library to a single thread so parallelism comes from
the worker *processes*, not nested thread pools. Two layers cooperate:

* :func:`numeric_thread_env` returns the env-var overrides; the pool exports
  them in the parent before spawning, and ``spawn`` children inherit them, so
  OpenMP/BLAS read the cap at import time inside each worker.
* :func:`limit_worker_threads` is a belt-and-suspenders runtime guard a worker
  enters around its consume loop, in case a library was imported before the env
  cap took effect.

Both layers are scoped to the match worker pool. Single-process commands
(``index build``, ``train-scorer``) never touch this module and keep their
multi-threaded numeric libraries.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from os import environ

from threadpoolctl import threadpool_limits

_SINGLE_THREAD: str = "1"

_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def numeric_thread_env() -> dict[str, str]:
    """Return the env-var overrides that pin every numeric lib to one thread.

    Returns:
        A mapping of each OpenMP/BLAS thread-count variable to ``"1"``,
        suitable for merging into ``os.environ`` before a ``spawn`` pool
        starts so its children inherit the single-thread cap.
    """
    return dict.fromkeys(_THREAD_ENV_VARS, _SINGLE_THREAD)


def pin_numeric_threads_in_env() -> None:
    """Export the single-thread cap into the current process environment.

    Called in the parent immediately before the match worker pool spawns its
    children. ``spawn`` workers inherit the environment, so OpenMP and BLAS see
    the cap when they are (re-)imported inside each worker.
    """
    environ.update(numeric_thread_env())


@contextmanager
def limit_worker_threads() -> Iterator[None]:
    """Cap numeric-library thread pools to one thread for the wrapped block.

    A runtime guard layered on top of :func:`pin_numeric_threads_in_env`: even
    if a library was imported before the env cap applied, entering this context
    forces its already-initialised thread pool down to a single thread for the
    duration of the worker's consume loop.
    """
    with threadpool_limits(limits=1):
        yield


__all__ = [
    "limit_worker_threads",
    "numeric_thread_env",
    "pin_numeric_threads_in_env",
]
