"""Regression evaluation against ground-truth pairs.

Phase 7 ships a minimal precision/recall + confusion-matrix evaluator
exposed via :func:`run_eval`; Phase 8 will expand this into the full
baseline/regression workflow.
"""

from pd_matcher.eval.ground_truth import EvalReport
from pd_matcher.eval.ground_truth import run_eval

__all__ = [
    "EvalReport",
    "run_eval",
]
