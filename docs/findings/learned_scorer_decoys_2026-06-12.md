# Learned-scorer decoy-negatives experiment — 2026-06-12

Issue #77, the #4 ORIGINAL design the research rounds dropped: train with sampled same-year-bucket non-matches as negatives. The production A/B (`docs/findings/learned_scorer_ab_2026-06-12.md`) showed the labeled-only learned model FAILS top-1 linkage (911 vs the weighted mean's 915 correct top picks) because it never trains on the rank-2..N decoy population it must out-rank at inference. This experiment augments the training set with those decoys and re-runs the A/B.

## Method

**Decoy harvest.** For each labeled-MATCH MARC the production `match_record` is run (under the weighted-mean scorer, exactly as eval pass A / `train-scorer` produce Evidence — the per-scorer Evidence is combiner-independent). Its `alternates` (up to 3 runners-up above the floor) already carry full Evidence at zero extra scoring cost; each is projected through the canonical `feature_row` and tagged a decoy negative. Realized counts:

- MARCs with a current `match` verdict: **922**
- of those resolved in the candidate pool: **922**
- MARCs yielding ≥1 decoy: **207**
- decoy negatives harvested: **267**
- alternates dropped as the true match: **4**; dropped as another labeled pair for that MARC: **0**

Many MARCs yield fewer than 3 alternates: under `year_window: 0` the floor and the same-year retrieval bound the runner-up pool, so the realized decoy count (267) is well below the 922×3 ceiling.

**Label-noise caveat.** A harvested alternate is assumed a non-match, but an unlabeled candidate could be a true duplicate registration. Any alternate whose `nypl_uuid` equals the labeled true match OR ANY labeled pair for that MARC is dropped, so known labels never poison the decoy set; the residual risk is an unlabeled true duplicate, accepted per the ticket as small.

**GroupKFold rationale.** Decoys and the positive from the same MARC share near-identical features; a random split would leak a MARC's positive into a fold that also holds its decoys, inflating OOF. GroupKFold by `marc_control_id` forces every row of one MARC into one fold, so the OOF numbers are honest under the augmented population.

- **Feature count**: 53 (production `feature_names()`; the persisted artifact's contract matches inference exactly)
- **Labeled rows**: 1434 (922 pos / 512 neg); **decoy rows**: 267; **total**: 1701
- **Cross-validation**: 5-fold GroupKFold by `marc_control_id`, locked hyperparameters (max_depth=3, num_leaves=8, min_data_in_leaf=10, lambda_l2=1.0, n_estimators=200, class_weight=balanced), random_state=20260612, deterministic (`n_jobs=1`)

## Grouped-OOF discrimination (baseline vs augmented)

Both models use identical locked hyperparameters, seed, and 5-group GroupKFold folds. The **full** rows measure each model on its own population (NOT comparable across models — the augmented population includes decoys the baseline never sees). The **labeled-rows-only** slice restricts the augmented OOF predictions to the labeled rows, the only apples-to-apples comparison against the labeled-only model.

| model | slice | rows | pos | OOF AUC | OOF PR-AUC | OOF best-F1 | at |
|:---|:---|---:|---:|---:|---:|---:|---:|
| labeled-only | full | 1434 | 922 | 0.9935 | 0.9965 | 0.9712 | 0.50 |
| augmented | full | 1701 | 922 | 0.9820 | 0.9791 | 0.9479 | 0.55 |
| labeled-only | labeled rows | 1434 | 922 | 0.9935 | 0.9965 | 0.9712 | 0.50 |
| augmented | labeled rows | 1434 | 922 | 0.9904 | 0.9948 | 0.9636 | 0.55 |

**Augmented minus labeled-only on the labeled-rows slice:** AUC -0.0031, best-F1 -0.0076 (same labeled rows, grouped OOF).

## A/B result — top-1 linkage (pass A)

The weighted-mean reference is unchanged (`/tmp/ab_weighted.json`): 915/922 correct top picks, F1 0.99349. The labeled-only learned model from the prior A/B scored 911/922. The decoy model below is evaluated by `pd-matcher eval --scorer learned` against the freshly-persisted `caches/learned_scorer.*`.

_Caveat (unchanged from the prior A/B):_ pass A on labeled MARCs is still partially train-set-flavored for the labeled rows, and pass-B AUC near 1.0 on labeled pairs remains uninformative. The decoy population at rank-2..N is now in-distribution, but the gate metric is pass-A top-1 vs the weighted reference.

| scorer | correct top / 922 | precision | recall | F1 |
|:---|---:|---:|---:|---:|
| weighted_mean (reference) | 915 | 0.99457 | 0.99241 | 0.99349 |
| learned (labeled-only, prior A/B) | 911 | 0.98914 | 0.98807 | 0.98861 |
| learned (decoy-augmented) | 913 | 0.99564 | 0.99024 | 0.99293 |

## Decision against the #77 gate

- **Gate 1 — top-1 F1 ≥ 0.99349 (weighted reference):** **FAIL, narrowly** — decoy-model F1 0.99293 (correct top 913/922 vs the weighted mean's 915; precision is now HIGHER than the weighted mean, 0.99564 vs 0.99457, with recall trailing).

- **Gate 2 — grouped-OOF labeled-rows slice not degraded vs labeled-only:** **FAIL** (AUC -0.0031, best-F1 -0.0076 on the labeled rows).

- **Gate 3 — throughput:** **PASS by construction.** Decoys change training only; inference is the identical per-candidate `Booster.predict`, so throughput is unchanged from the labeled-only learned model (85% of weighted mean in the prior A/B).

**Programmatic verdict: IMPROVED-BUT-SHORT** — the decoy model's 913 correct top picks beat the labeled-only model's 911 but stay below the weighted mean's 915.

## Interpretation and the next lever

The decoy mechanism works: +2 correct top picks from only **267** decoy rows. The harvest was ~10x thinner than the ticket anticipated (922 × 3 ≈ 2,800 ceiling) because `MatchResult.alternates` only carries runners-up **above the floor**, and under `year_window: 0` most MARCs have no above-floor runner-up at all (only 207 of 922 MARCs yielded any decoy). The model improved its ranking exactly where it saw decoys and stayed blind elsewhere.

The obvious next iteration: harvest **below-floor decoys** — score each labeled-MATCH MARC's full retrieved candidate set and take the top-k non-true candidates regardless of floor (k≈3-5), yielding ~10x the decoy mass over all 922 MARCs rather than 207. Gate 2's small labeled-rows degradation (−0.0076 best-F1) also deserves a class-weight or decoy-downweighting look in the same round: decoys should teach the ranking boundary without distorting the labeled-pair decision surface. Both are parameter/data changes to this script; no src/ changes.

The default stays `weighted_mean` either way until Gate 1 passes.

## Artifact + reproduction

The final Booster is fit on ALL augmented rows with the locked hyperparameters and persisted via the production `save_learned_model` to `caches/learned_scorer.{txt,msgpack}`, OVERWRITING the labeled-only artifact. Both files are gitignored and re-derivable: `pdm run pd-matcher train-scorer` rebuilds the labeled-only model; rerunning this script rebuilds the decoy-augmented one.

```
pdm run python scripts/learned_scorer_decoys.py \
    > docs/findings/learned_scorer_decoys_2026-06-12.md
pdm run pd-matcher eval --index caches/cce.lmdb --scorer learned \
    --report /tmp/ab_learned_decoys.json
```
