# Platt calibrator first-fit — 2026-06-07

First-ever Platt calibration of the production combiner against the labeled vault (issue #70). Prior to this fit the matcher ran with `calibrator=None`, so `combined.calibrated = combined.raw / 100` (a linear pass-through). This run fits a real sigmoid against the raw weighted-mean scores from the resolved vault pairs and writes the result to the production calibrator cache.

## 1. Training corpus

- **Vault**: `data/label_vault.jsonl`
- **Candidate pool**: `data/candidates`
- **Index**: `caches/cce.lmdb`
- **Positives** (verdict=`match`): 836
- **Negatives** (verdict=`no_match`): 362
- **Skipped (MARC missing from pool)**: 0
- **Skipped (CCE missing from index)**: 0

- **Positives raw score**: min=40.59 mean=88.00 max=100.00
- **Negatives raw score**: min=16.67 mean=54.59 max=95.27

## 2. Fitted Platt calibrator

- **`a`** (slope): `-0.165980`
- **`b`** (intercept): `11.001523`
- **`n_positive`**: 836
- **`n_negative`**: 362
- **`trained_at`**: `2026-06-07T15:39:02.672718+00:00`
- **Persisted to**: `caches/calibrator.msgpack`

## 3. Sanity-check probe table

Maps a representative raw weighted-mean score in `[0, 100]` to the calibrated probability returned by `calibrate(raw, calibrator)`. A well-formed calibrator is monotone-increasing: higher raw means higher probability. If this table is non-monotone or inverted the fit is broken.

| raw | calibrated |
|---:|---:|
| 50.0 | 0.0628 |
| 60.0 | 0.2606 |
| 65.0 | 0.4470 |
| 70.0 | 0.6496 |
| 75.0 | 0.8095 |
| 80.0 | 0.9069 |
| 90.0 | 0.9809 |
| 100.0 | 0.9963 |

## 4. Regression result under the new calibrator

Wired the calibrator through `run_eval` (`src/pd_matcher/eval/ground_truth.py`,
`src/pd_matcher/cli.py`, `tests/regression/test_regression.py`,
`tests/regression/update_baseline.py`) so the eval would actually consume
the cached file rather than running with `calibrator=None`.

| metric | pre (linear pass-through) | post (Platt) | Δ |
|:---|---:|---:|---:|
| precision | 0.99438 | 0.99286 | −0.00152 |
| recall | 0.84689 | **0.83134** | **−0.01555** |
| auc_roc | 0.94577 | 0.94577 | −0.00000 (monotone, ranks unchanged) |
| average_precision | 0.97641 | 0.97641 | +0.00000 |

## 5. Verdict — don't land

Recall dropped 1.55 pp; the probe table at §3 already predicted this — a
raw score of 70 maps to calibrated **0.6496** under Platt (vs **0.70** under
the prior linear pass-through), so the 0.70 floor catches *more* legitimate
matches than before, not fewer. The mid-band suppression makes the #20
floor-driven recall failure worse rather than better.

Likely root cause: the negative class skews high (mean raw 54.59, max
95.27) because the labeler tends to surface borderline cases, so Platt
reads the 60–70 raw band as "ambiguous" rather than "probable match" and
shrinks confidence in that range. The naive linear pass-through actually
fits this corpus's reality better than a sigmoid would.

**Action**: removed `caches/calibrator.msgpack` from the local cache (it
was never committed; `caches/` is gitignored). The `fit-calibrator`
script + the `run_eval` wiring are kept on the branch as durable
infrastructure for future calibrator attempts (e.g. after a scorer
change, or when the corpus is large enough that the borderline-negative
skew evens out).

**Implication for #71** (floor sweep): the blocker on #70 is lifted with
this negative result. The floor sweep should proceed against the linear
pass-through, not the Platt calibrator.
