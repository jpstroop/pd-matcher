# Learned-scorer held-out top-1 (k-fold through pipeline) — 2026-06-13

Issue #80. The FIRST confound-free top-1 comparison of the learned matcher against the weighted mean. Production `train-scorer` fits on ALL trainable vault pairs and `eval` then scores top-1 over the WHOLE vault, so the learned model grades itself on memorized pairs — its round-2 pass-A 919/922 (`docs/findings/learned_scorer_decoys_round2_2026-06-13.md`) is an UPPER BOUND, and the only leakage-free signal there (a grouped-OOF rank PROXY at 911/922) sits below the weighted mean's honest 915/922. This script removes the confound by construction: every evaluated MARC is scored by a fold-model that never trained on it.

## Method

**GroupKFold by `marc_control_id`, 5 folds.** The labeled-MATCH MARCs are partitioned into folds with each MARC as its own group, so every MARC — its labeled pairs AND any decoys harvested from it — lives in exactly one fold. There is no path for a MARC's own rows to leak into the model that grades it. Deterministic for a fixed MARC order and seed (`random_state=20260613`).

**Per held-out fold H, train from scratch on the other four folds.** The fold's training matrix is: every non-`unsure` vault pair (match + no_match) whose `marc_control_id` is a TRAIN-fold MARC, via the canonical `feature_row`; PLUS below-floor decoys — the top `5` non-true candidates per TRAIN-fold match-MARC, harvested by `match_record` under `min_combined_score=0.0` and `top_k=21` (the full ranked candidate set, no floor cull). Decoys are NEVER harvested from held-out MARCs. This reproduces the round-2 best config: decoy `sample_weight` w=0.5, locked hyperparameters (max_depth=3, num_leaves=8, min_data_in_leaf=10, lambda_l2=1.0, n_estimators=200, class_weight=balanced). The fold Booster is wrapped directly as the production `LearnedCombiner` (no disk round-trip; the production `caches/learned_scorer.*` artifact is left untouched).

**Held-out top-1 eval for fold H.** For each labeled-MATCH MARC in H, the production `match_record` runs with the FOLD model as combiner over the MARC's FULL retrieved candidate set (the configured floor applied exactly as production does). Top-1 is correct when the best pick's `nypl_uuid` equals the gold uuid. **The model-never-saw-the-evaluated-MARC guarantee is what makes this honest** — unlike pass-A eval, no evaluated pair was a training row.

**Weighted-mean reference, inline.** The weighted mean is deterministic and untrained, so its honest top-1 is its standard number. It is rerun inline over the EXACT same match-MARC set, fold by fold, so the denominator matches the learned column to the MARC.

- **Feature count**: 53 (production `feature_names()`)
- **Folds**: 5; held-out MARCs per fold: 185, 185, 184, 184, 184 (total 922)
- **Train-side rows (summed over folds)**: labeled 6248, decoys 18428 (each fold trains on ~4/5 of the labeled vault plus its own decoy harvest)

## Headline — honest held-out top-1

Both columns are over the same match-MARC set; the weighted column is rerun inline so the denominators are identical.

| scorer | correct top / 922 | precision | recall | F1 |
|:---|---:|---:|---:|---:|
| weighted_mean (honest, inline) | 918 | 0.99783 | 0.99566 | 0.99674 |
| learned (honest, k-fold held-out) | 896 | 0.99445 | 0.97180 | 0.98300 |

## Context — prior CONFOUNDED learned numbers (over the whole vault)

These were all measured with the learned model graded on pairs it trained on; they are upper bounds, shown for honest-vs-confounded contrast. The weighted mean has no training, so its 915/922 is already honest.

| measurement | correct top / 922 | leakage-free? |
|:---|---:|:---|
| weighted_mean (known full-vault) | 915 | yes (untrained) |
| learned labeled-only (pass A) | 911 | no (trained on eval pairs) |
| learned round-2 decoys (pass A) | 919 | no (trained on eval pairs) |
| learned round-2 grouped-OOF rank PROXY | 911 | partial (per-row proxy, not full top-1) |

## Decision

- **ADOPT-CANDIDATE** if honest learned top-1 clears weighted by more than 2 MARCs (recommend flipping the default).
- **PARITY** if within ±2 MARCs (noise).
- **HONEST-LOSS** if clearly below weighted.

**Verdict: HONEST-LOSS.** The honest held-out gap is learned minus weighted = -22 MARCs (896 vs 918), F1 -0.01375.

The learned matcher loses honestly to the weighted mean on unseen MARCs — the pass-A lead was entirely memorization. The weighted mean stays the production default; the learned artifact's value is the #76 audit queue, not top-1 linkage.

## Reproduction

```
pdm run python scripts/learned_scorer_heldout.py \
    > docs/findings/learned_scorer_heldout_2026-06-13.md
```

This script touches nothing under `src/` or `data/`, overwrites no artifact, and is deterministic (fixed seed, `n_jobs=1`).
