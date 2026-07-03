# Platt calibrator refit — 2026-07-03

Refit of the weighted-mean combiner's Platt probability calibrator against the current labeled vault (issue #70), and the durability fix that motivated it. This supersedes the [2026-06-07 first-fit](fit_calibrator_2026-06-07.md), whose verdict was *don't land*. The corpus has since roughly doubled and its negatives now separate cleanly from its positives, so the mid-band suppression that sank the earlier attempt no longer applies: this fit is landed as the recommended weighted-arm calibrator.

## 0. Why refit — the durability story

Three facts came together:

- The 2026-06-07 fit was deliberately **not landed** (recall dropped 1.55 pp; the negative class skewed high, so the sigmoid read the 60–70 raw band as ambiguous). Its artifact was removed from the local cache.
- `caches/` is gitignored, so the calibrator artifact is **never committed** — it lives only in a working cache. Any later cache wipe therefore leaves the weighted arm with *no* artifact.
- With no artifact the weighted arm silently falls back to `calibrated = raw / 100`. There is no warning. So every weighted-arm run between 2026-06-07 and this refit was uncalibrated, whether or not that was intended.

The silent fallback is the real hazard: an operator setting `--min-score 90` on an uncalibrated run is filtering on raw score, not on 90% confidence, with nothing in the output to say so. Making that non-silent (a logged notice, or a first-class `fit-calibrator` command that leaves a provenance breadcrumb) is filed as **#117**.

## 1. Training corpus

- **Vault**: `data/training/label_vault.jsonl`
- **Candidate pool**: `data/candidates`
- **Index**: `caches/cce.lmdb`
- **Positives** (verdict=`match`): 1086
- **Negatives** (verdict=`no_match`): 848
- **Skipped (MARC missing from pool)**: 0
- **Skipped (CCE missing from index)**: 0

- **Positives raw score**: min=25.93 mean=84.34 max=100.00
- **Negatives raw score**: min=0.00 mean=22.50 max=78.57

The corpus is ~1.6× the size of the 2026-06-07 fit (1,934 resolved pairs vs 1,198), and — decisively — the negatives now separate. The earlier fit's negatives had mean raw **54.59** / max **95.27**; this fit's negatives have mean raw **22.50** / max **78.57**. The borderline-negative skew the 2026-06-07 doc named as the root cause of its recall loss has evened out, which is why the same procedure now produces a usable calibrator instead of a mid-band-suppressing one.

## 2. Fitted Platt calibrator

- **`a`** (slope): `-0.135969`
- **`b`** (intercept): `6.989077`
- **`n_positive`**: 1086
- **`n_negative`**: 848
- **`trained_at`**: `2026-07-03T22:36:33.638047+00:00`
- **Persisted to**: `caches/calibrator.msgpack`

## 3. Sanity-check probe table

Maps a representative raw weighted-mean score in `[0, 100]` to the calibrated probability returned by `calibrate(raw, calibrator)`. A well-formed calibrator is monotone-increasing: higher raw means higher probability. If this table is non-monotone or inverted the fit is broken.

| raw | calibrated |
|---:|---:|
| 50.0 | 0.4525 |
| 60.0 | 0.7630 |
| 65.0 | 0.8640 |
| 70.0 | 0.9261 |
| 75.0 | 0.9612 |
| 80.0 | 0.9799 |
| 90.0 | 0.9948 |
| 100.0 | 0.9987 |

Contrast with the 2026-06-07 probe table, which mapped raw 70 to **0.6496** — *below* the linear pass-through's 0.70, so the 0.70 floor caught more matches uncalibrated than calibrated. This fit maps raw 70 to **0.9261**, above the pass-through, because the cleaner negative separation lets the sigmoid put the decision boundary lower. A `--min-score 90` cut now lands around raw 65 rather than punishing the mid-band.

## 4. Provenance and refit procedure

The artifact is fit out-of-band — nothing in `index` or `match` builds it. The frozen proof script is `scripts/fit_calibrator.py`: it resolves each non-`unsure` vault entry to its `(MARC, CCE)` pair, scores it with the production weighted-mean combiner, partitions the raw scores into positives (`verdict=match`) and negatives (`verdict=no_match`), and fits `a` and `b` by Newton iteration, writing the result to `caches/calibrator.msgpack`. To refit after a scorer change or further vault growth:

```bash
pdm run python scripts/fit_calibrator.py
```

Once the artifact exists, `pd-matcher match` and `pd-matcher eval` auto-load `caches/calibrator.msgpack` (from the index's parent directory) and the weighted arm emits calibrated probabilities. A first-class `fit-calibrator` subcommand that replaces the raw script — and surfaces the silent-fallback state — is tracked at **#117**.
