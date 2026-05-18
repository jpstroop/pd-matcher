"""Parallel worker pool, producer/writer processes, and progress reporter.

The public entry point is :func:`run_match`; everything else in this
package is exposed for testing.
"""

from pd_matcher.workers.pool import RunReport
from pd_matcher.workers.pool import run_match

__all__ = [
    "RunReport",
    "run_match",
]
