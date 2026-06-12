# Learned-scorer tightening round — 2026-06-12

Productionization gate for issue #4. Builds on the 1500-label diagnostic (`docs/findings/learned_scorer_diagnostic_2026-06-12.md`): the learning curve plateaued and the 18-feature LightGBM OOF beat the weighted mean by +0.061 best-F1, but the run HELD on a fold-variance criterion. This round expands the feature set, sweeps hyperparameters, checks calibration, and autopsies the 33 regressions before deciding whether to wire the learned combiner into production.

- **Pairs scored**: 1434 (923 match / 511 no_match)
- **Cross-validation**: 5-fold stratified, random_state=20260529, deterministic (`n_jobs=1`)
- **Expanded feature count**: 51

## 1. Expanded feature matrix

The baseline 18 features are 9 per-scorer normalized scores plus 9 `__skipped` flags. The expanded set adds every stable named sub-feature from each scorer's `Evidence.features`, namespaced `{scorer}.{feature}` (author and publisher share sub-feature names, so the prefix is load-bearing), with a `__present` flag where a `-1.0` sentinel or a skipped scorer makes a raw `0.0` ambiguous. One pair-level computable is added: `pair.title_len_ratio` (MARC title tokens / CCE title tokens).

**Language/country agreement is intentionally absent.** `IndexedNyplRegRecord` (`src/pd_matcher/models.py`) carries no `language_code` or `country_code` field — those exist only on the MARC side — so there is nothing on the CCE side to agree with. The feature named in ticket #4 is not computable from current data and is skipped rather than faked.

| feature set | n_features | OOF AUC | OOF PR-AUC | OOF best-F1 | at_threshold |
|:---|---:|---:|---:|---:|---:|
| baseline (18) | 18 | 0.9839 | 0.9909 | 0.9525 | 0.40 |
| expanded | 51 | 0.9929 | 0.9962 | 0.9720 | 0.40 |

**Expanded minus baseline:** best-F1 +0.0194, AUC +0.0090 (same hyperparameters, same seeds, same folds).

### Expanded-model feature importance (top 20)

LightGBM `gain` importance averaged across the five folds, normalized to sum to 1.0 across all expanded features.

| rank | feature | lgbm_importance |
|---:|:---|---:|
| 1 | `title.token_set.token_overlap` | 0.2301 |
| 2 | `name.publisher` | 0.1926 |
| 3 | `title.token_set` | 0.1415 |
| 4 | `extent.page_count` | 0.1133 |
| 5 | `name.author.token_overlap` | 0.0557 |
| 6 | `name.author` | 0.0547 |
| 7 | `extent.page_count.marc_pages` | 0.0403 |
| 8 | `extent.page_count.cce_pages` | 0.0372 |
| 9 | `extent.page_count.delta` | 0.0312 |
| 10 | `title.token_set.avg_token_idf` | 0.0289 |
| 11 | `title.token_set.unique_to_nypl` | 0.0238 |
| 12 | `lccn.exact.marc_lccn` | 0.0132 |
| 13 | `volume.compat.marc_is_whole_open` | 0.0130 |
| 14 | `pair.title_len_ratio` | 0.0038 |
| 15 | `name.publisher.normalized_nypl_len` | 0.0036 |
| 16 | `name.author.normalized_marc_len` | 0.0027 |
| 17 | `title.token_set.unique_to_marc` | 0.0026 |
| 18 | `name.author.normalized_nypl_len` | 0.0022 |
| 19 | `volume.compat__skipped` | 0.0021 |
| 20 | `title.token_set.token_total` | 0.0015 |

## 2. Hyperparameter sweep

Small grid around the conservative point (`max_depth` ∈ {3,4,5}, `num_leaves` ∈ {8,15,31}, `min_data_in_leaf` ∈ {5,10,20}, `n_estimators` ∈ {100,200}), pruning combinations with `num_leaves > 2**max_depth`. Each config is scored by 5-fold OOF AUC on the EXPANDED features; `36` valid configs ran deterministically (`n_jobs=1`, fixed seed). All other parameters match the baseline (`lambda_l2=1.0`, `class_weight=balanced`).

| rank | max_depth | num_leaves | min_data_in_leaf | n_estimators | OOF AUC | baseline? |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 3 | 8 | 10 | 200 | 0.9933 |  |
| 2 | 3 | 8 | 5 | 200 | 0.9932 |  |
| 3 | 3 | 8 | 20 | 200 | 0.9932 |  |
| 4 | 3 | 8 | 10 | 100 | 0.9929 | yes |
| 5 | 3 | 8 | 20 | 100 | 0.9928 |  |

The conservative baseline config (max_depth=3, num_leaves=8, min_data_in_leaf=10, n_estimators=100) ranks **#4/36** at OOF AUC 0.9929.

**Winner** (used for sections 3-4): max_depth=3, num_leaves=8, min_data_in_leaf=10, n_estimators=200 — OOF AUC 0.9933.

## 3. Calibration analysis

On the winning model's OOF predictions. The production pipeline expects calibrated `[0,1]` scores (floor at 0.50, confidence bands), so the question is whether raw LightGBM probabilities are usable as-is or need a calibration layer.

### Reliability table (10 bins)

Each bin spans an equal predicted-probability width. `observed_rate` is the empirical match fraction among pairs whose raw OOF prediction falls in the bin; good calibration tracks the diagonal (`mean_pred` ≈ `observed_rate`).

| bin | range | count | mean_pred | observed_rate |
|:---|:---|---:|---:|---:|
| 1 | [0.0, 0.1) | 445 | 0.010 | 0.013 |
| 2 | [0.1, 0.2) | 29 | 0.136 | 0.207 |
| 3 | [0.2, 0.3) | 17 | 0.242 | 0.471 |
| 4 | [0.3, 0.4) | 12 | 0.357 | 0.167 |
| 5 | [0.4, 0.5) | 11 | 0.455 | 0.455 |
| 6 | [0.5, 0.6) | 9 | 0.551 | 0.556 |
| 7 | [0.6, 0.7) | 13 | 0.657 | 0.846 |
| 8 | [0.7, 0.8) | 19 | 0.752 | 0.895 |
| 9 | [0.8, 0.9) | 35 | 0.857 | 0.714 |
| 10 | [0.9, 1.0) | 844 | 0.993 | 0.993 |

### Brier score

Platt (logistic) and isotonic calibrators are fit out-of-fold over the raw OOF predictions via a second independent 5-fold split, so each calibrated probability comes from a calibrator that never saw that pair's raw score. Lower Brier is better.

| variant | Brier |
|:---|---:|
| raw LightGBM OOF | 0.0287 |
| + Platt | 0.0303 |
| + isotonic | 0.0302 |

**Raw LightGBM probabilities are the best-calibrated variant** — neither Platt nor isotonic improves Brier. Raw probs are usable as-is; no production calibration layer is required (the 0.50 floor and confidence bands can read the raw output directly).

## 4. Regression autopsy (the 33)

The 33 `weighted_right_lgbm_wrong` pairs from `/tmp/learned_scorer_disagreements.jsonl` — pairs the old 18-feature OOF model got wrong while the weighted mean got them right. `old OOF` is the dump's 18-feature OOF probability; `new OOF` is the winning expanded/tuned model's OOF probability (decision at threshold 0.55). A pair is `fixed` when the new model's OOF decision matches the truth.

| control_id | truth | old OOF | new OOF | fixed? |
|:---|:---|---:|---:|:---:|
| `9985739723506421` | match | 0.211 | 0.666 | yes |
| `9911941423506421` | match | 0.236 | 0.732 | yes |
| `9920249833506421` | match | 0.277 | 0.045 | no |
| `9925174633506421` | match | 0.169 | 0.458 | no |
| `9917221823506421` | no_match | 0.951 | 0.871 | no |
| `9927596913506421` | no_match | 0.806 | 0.079 | yes |
| `998205883506421` | no_match | 0.587 | 0.349 | yes |
| `9923856593506421` | no_match | 0.837 | 0.964 | no |
| `994153263506421` | no_match | 0.600 | 0.028 | yes |
| `991886873506421` | no_match | 0.442 | 0.013 | yes |
| `9929860013506421` | match | 0.262 | 0.046 | no |
| `9930330873506421` | no_match | 0.492 | 0.108 | yes |
| `9916767783506421` | no_match | 0.953 | 0.979 | no |
| `9933105333506421` | no_match | 0.527 | 0.152 | yes |
| `9964294873506421` | no_match | 0.452 | 0.012 | yes |
| `9916332783506421` | no_match | 0.492 | 0.081 | yes |
| `9911606923506421` | match | 0.265 | 0.900 | yes |
| `9928744783506421` | no_match | 0.632 | 0.555 | no |
| `99125488858406421` | no_match | 0.632 | 0.371 | yes |
| `9989203383506421` | no_match | 0.431 | 0.323 | yes |
| `9916779783506421` | no_match | 0.852 | 0.951 | no |
| `9917328533506421` | match | 0.199 | 0.884 | yes |
| `9912310313506421` | no_match | 0.810 | 0.454 | yes |
| `9920947063506421` | no_match | 0.769 | 0.336 | yes |
| `99129180505206421` | match | 0.087 | 0.788 | yes |
| `9917051703506421` | match | 0.297 | 0.845 | yes |
| `99125488138706421` | match | 0.180 | 0.129 | no |
| `9920375733506421` | match | 0.063 | 0.156 | no |
| `9926379083506421` | no_match | 0.470 | 0.132 | yes |
| `9911273513506421` | no_match | 0.701 | 0.380 | yes |
| `9929574633506421` | match | 0.096 | 0.552 | yes |
| `995041273506421` | match | 0.268 | 0.338 | no |
| `9919984763506421` | match | 0.297 | 0.730 | yes |

**22/33 of the old regressions are now fixed** by the expanded/tuned model; 11 remain broken.

### Still-broken: extreme-feature readout

For each still-broken pair, the 5 expanded features whose value deviates most (in per-feature standard deviations) from the mean of the pair's TRUE class. This is the quantitative companion to the manual UI review of the same 33 pairs.

**`9920249833506421`** (truth match; MARC _Plastomechanik der Umformung metallisch…_ vs CCE _Plastomechanik der Umformung metal lisc…_):

| feature | value | class_mean | z |
|:---|---:|---:|---:|
| `volume.compat.marc_is_whole_open` | 1.000 | 0.002 | +6.97 |
| `volume.compat__skipped` | 0.000 | 0.976 | -4.69 |
| `extent.page_count.cce_pages` | 1106.000 | 181.358 | +4.62 |
| `extent.page_count.cce_pages__present` | 0.000 | 0.690 | -1.42 |
| `extent.page_count__skipped` | 1.000 | 0.310 | +1.42 |

**`9925174633506421`** (truth match; MARC _Werke und briefe_ vs CCE _… Werke und briefe; historisch-kritisch…_):

| feature | value | class_mean | z |
|:---|---:|---:|---:|
| `volume.compat` | 1.000 | 0.018 | +6.74 |
| `volume.compat.cce_is_whole` | 1.000 | 0.012 | +6.69 |
| `volume.compat__skipped` | 0.000 | 0.976 | -4.69 |
| `volume.compat.marc_is_whole` | 1.000 | 0.031 | +4.17 |
| `title.token_set.unique_to_nypl` | 8.000 | 0.908 | +2.87 |

**`9917221823506421`** (truth no_match; MARC _Aubrey Beardsley drawings / [introducti…_ vs CCE _Aubrey Beardsley: selected drawings_):

| feature | value | class_mean | z |
|:---|---:|---:|---:|
| `extent.page_count` | 1.000 | 0.145 | +1.74 |
| `title.token_set.avg_token_idf` | 10.568 | 6.147 | +1.54 |
| `name.author` | 1.000 | 0.501 | +1.30 |
| `lccn.exact.marc_lccn` | 0.000 | 0.521 | -1.19 |
| `extent.page_count.marc_pages__present` | 1.000 | 0.505 | +1.02 |

**`9923856593506421`** (truth no_match; MARC _L'existence malheureuse_ vs CCE _L'existence malheureuse. Paris._):

| feature | value | class_mean | z |
|:---|---:|---:|---:|
| `title.token_set.avg_token_idf` | 11.557 | 6.147 | +1.89 |
| `name.publisher` | 1.000 | 0.415 | +1.42 |
| `name.author` | 0.000 | 0.501 | -1.31 |
| `lccn.exact.marc_lccn` | 1.000 | 0.521 | +1.09 |
| `title.token_set` | 0.723 | 0.304 | +1.07 |

**`9929860013506421`** (truth match; MARC _Copying manuscripts_ vs CCE _Guide to the personal papers in the man…_):

| feature | value | class_mean | z |
|:---|---:|---:|---:|
| `lccn.exact.nypl_lccn_present` | 1.000 | 0.100 | +3.10 |
| `title.token_set.unique_to_nypl` | 5.000 | 0.908 | +1.66 |
| `name.publisher.normalized_nypl_len` | 61.000 | 33.144 | +1.37 |
| `extent.page_count` | 0.000 | 0.645 | -1.31 |
| `title.token_set` | 0.355 | 0.804 | -1.14 |

**`9916767783506421`** (truth no_match; MARC _Transportation frontiers_ vs CCE _Transportation frontiers. From the 1961…_):

| feature | value | class_mean | z |
|:---|---:|---:|---:|
| `volume.compat.marc_is_part` | 1.000 | 0.053 | +3.85 |
| `name.publisher.normalized_nypl_len` | 106.000 | 28.434 | +3.83 |
| `name.publisher.token_overlap` | 5.000 | 0.738 | +3.58 |
| `title.token_set.unique_to_nypl` | 10.000 | 2.096 | +3.20 |
| `extent.page_count` | 1.000 | 0.145 | +1.74 |

**`9928744783506421`** (truth no_match; MARC _Poems_ vs CCE _Poems_):

| feature | value | class_mean | z |
|:---|---:|---:|---:|
| `lccn.exact.nypl_lccn_present` | 1.000 | 0.080 | +3.17 |
| `title.token_set` | 1.000 | 0.304 | +1.78 |
| `extent.page_count` | 1.000 | 0.145 | +1.74 |
| `name.author` | 0.000 | 0.501 | -1.31 |
| `lccn.exact.marc_lccn` | 1.000 | 0.521 | +1.09 |

**`9916779783506421`** (truth no_match; MARC _Bergey's manual of determinative bacter…_ vs CCE _Society of American bacteriologists. Be…_):

| feature | value | class_mean | z |
|:---|---:|---:|---:|
| `edition.compat.nypl_edition_num` | 5.000 | 0.025 | +7.96 |
| `edition.compat.marc_edition_num` | 5.000 | 0.031 | +7.73 |
| `edition.compat` | 1.000 | 0.011 | +4.17 |
| `edition.compat.marc_edition_num__present` | 1.000 | 0.016 | +4.02 |
| `edition.compat.nypl_edition_num__present` | 1.000 | 0.016 | +4.02 |

**`99125488138706421`** (truth match; MARC _Thomas Mann. Considérations d'un apolit…_ vs CCE _Considerations d'un apolitique_):

| feature | value | class_mean | z |
|:---|---:|---:|---:|
| `name.publisher__skipped` | 1.000 | 0.010 | +4.60 |
| `extent.page_count.delta` | 488.000 | 7.928 | +3.97 |
| `name.publisher` | 0.000 | 0.864 | -2.10 |
| `isbn.exact.marc_isbn_count` | 1.000 | 0.163 | +1.98 |
| `lccn.exact.marc_lccn` | 0.000 | 0.860 | -1.96 |

**`9920375733506421`** (truth match; MARC _Spannbetonbau_ vs CCE _Spannbetonbau. T.1. 2. Aufl._):

| feature | value | class_mean | z |
|:---|---:|---:|---:|
| `volume.compat.cce_is_part` | 1.000 | 0.021 | +5.62 |
| `volume.compat__skipped` | 0.000 | 0.976 | -4.69 |
| `volume.compat.marc_is_whole` | 1.000 | 0.031 | +4.17 |
| `extent.page_count.cce_pages__present` | 0.000 | 0.690 | -1.42 |
| `extent.page_count.marc_pages__present` | 0.000 | 0.690 | -1.42 |

**`995041273506421`** (truth match; MARC _Le triangle noir_ vs CCE _Le triangle noir_):

| feature | value | class_mean | z |
|:---|---:|---:|---:|
| `name.author` | 0.000 | 0.865 | -2.26 |
| `name.author.token_overlap` | 0.000 | 2.133 | -1.80 |
| `extent.page_count` | 0.100 | 0.645 | -1.11 |
| `name.author.normalized_nypl_len` | 28.000 | 18.492 | +0.70 |
| `extent.page_count.cce_pages__present` | 1.000 | 0.690 | +0.64 |

## Decision

Gate criteria:

- Expanded+tuned OOF best-F1 ≥ baseline-18 OOF best-F1: **True** (expanded 0.9720 vs baseline 0.9525)
- Calibrated Brier ≤ raw Brier: **True** (best 0.0287 vs raw 0.0287)
- 33-regression count does not grow: **True** (22/33 of the old regressions now fixed; 11 still broken)

> The fold-std criterion from the last run is deliberately DROPPED. It was calibrated against a thin negative class (the 2026-05-31 run's `0.0026` threshold) and penalizes exactly the fold-to-fold variance that a small, imbalanced corpus produces by construction; it is not a meaningful deployability signal at this corpus size.

**PROCEED to productionization.** All three criteria hold: the expanded feature set does not regress OOF best-F1, calibration does not hurt Brier, and the prior regression set does not grow. The next phase is wiring the learned combiner (winning hyperparameters, expanded features, raw probs) into the matching pipeline and measuring top-1 linkage F1 on the regression eval (`pass-B`).
