<!-- fold 1: roc_auc=0.9859 pr_auc=0.9920 -->
<!-- fold 2: roc_auc=0.9860 pr_auc=0.9922 -->
<!-- fold 3: roc_auc=0.9875 pr_auc=0.9930 -->
<!-- fold 4: roc_auc=0.9877 pr_auc=0.9939 -->
<!-- fold 5: roc_auc=0.9731 pr_auc=0.9837 -->
# Learned-scorer diagnostic — 2026-06-12

## 1. Experimental setup

- **Pairs scored**: 1434 (923 match / 511 no_match)
- **Cross-validation**: 5-fold stratified, random_state=20260529
- **Model**: LightGBM binary classifier, hyperparameters:
    - `max_depth`: `3`
    - `num_leaves`: `8`
    - `min_data_in_leaf`: `10`
    - `lambda_l2`: `1.0`
    - `n_estimators`: `100`
    - `class_weight`: `balanced`
    - `objective`: `binary`
    - `verbose`: `-1`
    - `random_state`: `20260529`

- **ROC-AUC** across folds: mean=0.9840 std=0.0055
- **PR-AUC** across folds: mean=0.9910 std=0.0037

> The negative class is the binding constraint at this corpus size; fold-level numbers are directional, not deployable. Recompute at ~1500 labels.

## 2. Feature importance ranking

LightGBM `gain` importance averaged across the five folds, normalized to sum to 1.0 across all features. The `current_weight` column shows the matching.yaml weight for that scorer (skipped-flag features have no direct weight analogue and are marked `--`).

| rank | feature | lgbm_importance | current_weight |
|---:|:---|---:|---:|
| 1 | `title.token_set` | 0.3487 | 0.350 |
| 2 | `name.publisher` | 0.2605 | 0.100 |
| 3 | `extent.page_count` | 0.1787 | 0.050 |
| 4 | `name.author` | 0.1362 | 0.200 |
| 5 | `volume.compat__skipped` | 0.0375 | -- |
| 6 | `extent.page_count__skipped` | 0.0288 | -- |
| 7 | `edition.compat` | 0.0049 | 0.050 |
| 8 | `volume.compat` | 0.0026 | 0.050 |
| 9 | `name.publisher__skipped` | 0.0010 | -- |
| 10 | `name.author__skipped` | 0.0010 | -- |
| 11 | `title.token_set__skipped` | 0.0002 | -- |
| 12 | `edition.compat__skipped` | 0.0001 | -- |
| 13 | `lccn.exact` | 0.0000 | 0.100 |
| 14 | `isbn.exact` | 0.0000 | 0.000 |
| 15 | `year.delta` | 0.0000 | 0.100 |
| 16 | `year.delta__skipped` | 0.0000 | -- |
| 17 | `lccn.exact__skipped` | 0.0000 | -- |
| 18 | `isbn.exact__skipped` | 0.0000 | -- |

## 3. Per-feature SHAP contribution distributions

Mean absolute contribution, standard deviation, and a coarse direction label across all out-of-fold predictions. High `std` relative to `mean_abs` signals interaction effects: the feature pushes predictions in different directions in different contexts.

| feature | mean_abs | std | direction |
|:---|---:|---:|:---|
| `title.token_set` | 2.0078 | 2.3761 | bidirectional |
| `extent.page_count` | 1.6789 | 1.7552 | bidirectional |
| `name.publisher` | 1.3225 | 1.4285 | bidirectional |
| `name.author` | 1.2313 | 1.3231 | bidirectional |
| `extent.page_count__skipped` | 0.4401 | 0.4696 | bidirectional |
| `volume.compat__skipped` | 0.1769 | 0.4390 | positive |
| `edition.compat` | 0.1236 | 0.2637 | negative |
| `name.author__skipped` | 0.0229 | 0.0438 | negative |
| `volume.compat` | 0.0163 | 0.0575 | bidirectional |
| `title.token_set__skipped` | 0.0151 | 0.0282 | negative |
| `name.publisher__skipped` | 0.0088 | 0.0226 | bidirectional |
| `edition.compat__skipped` | 0.0007 | 0.0039 | negative |
| `isbn.exact` | 0.0000 | 0.0000 | inert |
| `year.delta` | 0.0000 | 0.0000 | inert |
| `year.delta__skipped` | 0.0000 | 0.0000 | inert |
| `lccn.exact__skipped` | 0.0000 | 0.0000 | inert |
| `isbn.exact__skipped` | 0.0000 | 0.0000 | inert |
| `lccn.exact` | 0.0000 | 0.0000 | inert |

## 4. Top-30 disagreement pairs

Pairs sorted by `|lgbm_pred - combined_score|` (descending). `verdict` is the human label. Use this as the worktable for the next round of weight inspection.

| rank | pair_id | marc_control_id | nypl_uuid | verdict | combined | lgbm | |delta| | marc_title | cce_title |
|---:|---:|:---|:---|:---|---:|---:|---:|:---|:---|
| 1 | 627 | `9990281473506421` | `5E1E18EC-7950-1014-8674-C1ED2059688C` | no_match | 0.940 | 0.019 | 0.921 | The Tudors | A |
| 2 | 607 | `992445023506421` | `AFF5E6E9-72DD-1014-B3A3-DDF21CF07E3D` | no_match | 0.838 | 0.010 | 0.828 | Anglo-Norman armory | ? |
| 3 | 261 | `994141873506421` | `D851443E-6C47-1014-802D-A968AD0E9A77` | no_match | 0.825 | 0.049 | 0.776 | Untitled 2 & 3 | Untitled |
| 4 | 1426 | `99131273843906421` | `983F38FC-6C6D-1014-965B-8464748AEC84` | no_match | 0.750 | 0.001 | 0.749 | Greek posters collection (processed) | Why? |
| 5 | 654 | `9926401183506421` | `0F3DC319-73D0-1014-A526-C29EA9D2D19F` | no_match | 0.750 | 0.001 | 0.749 | A Style of our own | The |
| 6 | 888 | `9929881443506421` | `F589566E-7454-1014-BDF7-E78324322810` | no_match | 0.750 | 0.003 | 0.747 | American business polices | This is it |
| 7 | 641 | `9927099793506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.004 | 0.746 | Friday Mount: first settlement at Holbrook and the south-we… | Where to? |
| 8 | 1407 | `9932024423506421` | `513A7200-7119-1014-B0E4-A14A79DE7F92` | no_match | 0.750 | 0.004 | 0.746 | The structural elements responsible for contraction in the … | Who is he? |
| 9 | 616 | `9911809583506421` | `0F3DC319-73D0-1014-A526-C29EA9D2D19F` | no_match | 0.750 | 0.005 | 0.745 | Graduation exercises, Bolinas School | The |
| 10 | 1432 | `9919097093506421` | `51F07EE1-6F10-1014-90C3-93BD933BF4A5` | no_match | 0.761 | 0.017 | 0.745 | Proceedings | Proceedings. |
| 11 | 79 | `9917021493506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.010 | 0.740 | Victorian photographs of famous men & fair women | Where to? |
| 12 | 903 | `9916882783506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.015 | 0.735 | Glossary of geology | Where to? |
| 13 | 114 | `994320373506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.015 | 0.735 | Chaucer's Constance and accused queens | Where to? |
| 14 | 631 | `9911753913506421` | `811C1A97-6CA4-1014-AFF1-C0B6E171A089` | no_match | 0.740 | 0.009 | 0.731 | Proceedings | Proceedings. |
| 15 | 921 | `9917014243506421` | `0F3DC319-73D0-1014-A526-C29EA9D2D19F` | no_match | 0.733 | 0.003 | 0.730 | Juvenal and declamtion | The |
| 16 | 883 | `9929729623506421` | `513A7200-7119-1014-B0E4-A14A79DE7F92` | no_match | 0.716 | 0.005 | 0.711 | The dead star | Who is he? |
| 17 | 797 | `9915783493506421` | `DE5AD668-6F5B-1014-A2B7-83BFD15252F6` | no_match | 0.754 | 0.044 | 0.710 | Three to be read | What to read. |
| 18 | 912 | `9915951973506421` | `DB7BFC82-728D-1014-8500-D70EB891A161` | no_match | 0.710 | 0.006 | 0.704 | Collected poems | Collected poems. |
| 19 | 292 | `9912111803506421` | `A870FAEB-7E9A-1014-A737-EA626E1B7465` | no_match | 0.708 | 0.005 | 0.703 | Cold war to détente | From the cold war to detente |
| 20 | 911 | `9933244693506421` | `513A7200-7119-1014-B0E4-A14A79DE7F92` | no_match | 0.707 | 0.004 | 0.703 | The Sacco-Vanzetti case | Who is he? |
| 21 | 791 | `9927048753506421` | `992C69E8-7176-1014-A92C-DCC6D41F8A68` | no_match | 0.748 | 0.048 | 0.700 | The next Australia | Australia |
| 22 | 1399 | `994420443506421` | `26BACDBB-6CCB-1014-A5FE-CCE70832EF77` | no_match | 0.700 | 0.002 | 0.698 | The opera-ballets of André Campra | Who? |
| 23 | 1342 | `9912104833506421` | `037207A4-71C6-1014-8134-C93186857334` | no_match | 0.700 | 0.002 | 0.698 | Beliefs and practices associated with Muslim pirs in two ci… | Where? |
| 24 | 891 | `9916748343506421` | `0F3DC319-73D0-1014-A526-C29EA9D2D19F` | no_match | 0.700 | 0.003 | 0.697 | History and cultivation of cotton and tobacco | The |
| 25 | 1031 | `9922810323506421` | `61DBF8C9-7673-1014-8449-EA9F584A306B` | no_match | 0.746 | 0.050 | 0.697 | II ̊Congresso internacional de historia de América | Panorama internacional de America. |
| 26 | 1366 | `992888713506421` | `3BFDE4F6-70AF-1014-98B5-BD1CA90B7FB2` | no_match | 0.700 | 0.004 | 0.696 | A sitter for a satyr | This in which. |
| 27 | 1100 | `9911719483506421` | `69DA871B-6D03-1014-999C-964E71FFACD8` | no_match | 0.698 | 0.003 | 0.695 | Fungi in oceans and estuaries | From this to that. |
| 28 | 1368 | `9912130613506421` | `B32D57BD-7950-1014-8674-C1ED2059688C` | no_match | 0.696 | 0.001 | 0.695 | Black elite | The will |
| 29 | 611 | `9917170793506421` | `3CC51FE1-70AF-1014-98B5-BD1CA90B7FB2` | no_match | 0.868 | 0.173 | 0.694 | Report and selected papers | Selecting annual report papers. |
| 30 | 1151 | `9949672653506421` | `642EB8A5-75F5-1014-96EA-B4CA74C85597` | no_match | 0.692 | 0.001 | 0.691 | El reino animal | O! |

## 5. Learning curve

Stratified-by-label subsamples drawn with a fixed seed (`random_state=20260529`); the existing 5-fold CV and hyperparameters run at each size. `mean_auc` is the mean out-of-fold ROC-AUC across folds, `std` its fold standard deviation, `pr_auc` the mean fold PR-AUC.

| requested_n | realized_n | mean_auc | std | pr_auc |
|---:|---:|---:|---:|---:|
| 500 | 500 | 0.9815 | 0.0069 | 0.9899 |
| 750 | 750 | 0.9835 | 0.0056 | 0.9911 |
| 1000 | 1000 | 0.9829 | 0.0045 | 0.9907 |
| 1250 | 1250 | 0.9862 | 0.0040 | 0.9925 |
| all | 1434 | 0.9840 | 0.0055 | 0.9910 |

**Verdict: PLATEAU** — AUC delta between the last two points is 0.0021; one fold-std of the larger point is 0.0055.

## 6. Head-to-head: LightGBM OOF vs weighted-mean combiner

Identical pair set, identical Evidence. The weighted-mean score is the production combiner's calibrated output (`combined_score`, deterministic — no CV); the LightGBM column is the out-of-fold probability from the 5-fold CV, so every pair is scored by a model that never saw it.

| scorer | AUC | PR-AUC | best_F1 | at_threshold |
|:---|---:|---:|---:|---:|
| weighted-mean | 0.9422 | 0.9677 | 0.8915 | 0.65 |
| LightGBM OOF | 0.9839 | 0.9909 | 0.9525 | 0.40 |

**Best-F1 delta (LightGBM − weighted-mean): +0.0611.** Issue #4's informal adoption bar is ~0.02 F1 points. Note: the authoritative bar is top-1 linkage F1 on the regression eval (`pass-B`), which requires wiring the learned combiner into the matching pipeline — a separate, next-phase task. The OOF best-F1 here is a per-pair classification proxy, not the linkage metric.

## 7. Category-sliced analysis

Categories are deliberately not model inputs. They are assigned by the labeler at verdict time and do not exist for unlabeled pairs, so a combiner trained on them could never run at inference; worse, they encode the verdict itself (e.g. `marc_whole_cce_part` is 85% `no_match`), so training on them is label leakage. They are used here only to slice evaluation results.

Per-category accuracy at each scorer's own best-F1 threshold (weighted-mean @ 0.65, LightGBM @ 0.40). `unsure` verdicts are already excluded by the feature matrix; only `match`/`no_match` pairs appear.

| category | n | weighted_acc | lgbm_acc | delta |
|:---|---:|---:|---:|---:|
| `marc_whole_cce_part` | 35 | 0.343 | 0.771 | +0.429 |
| `cce_whole_marc_part` | 3 | 0.667 | 1.000 | +0.333 |
| `translation` | 16 | 0.750 | 0.875 | +0.125 |
| `different_edition` | 4 | 1.000 | 0.500 | -0.500 |
| `ocr_confusion` | 16 | 0.875 | 1.000 | +0.125 |
| `same_title_different_work` | 6 | 0.833 | 0.667 | -0.167 |
| `generic_title` | 17 | 0.529 | 0.882 | +0.353 |

> 97 category-tag occurrences fall on scored (non-`unsure`) pairs; a pair tagged with multiple categories is counted under each. `unsure`-verdict pairs carrying tags are dropped by the feature matrix and absent above.

### LightGBM right, weighted-mean wrong (151 pairs, showing up to 25)

| marc_control_id | truth | weighted | lgbm | categories | note? |
|:---|:---|---:|---:|:---|:---|
| `9917021493506421` | no_match | 0.750 | 0.010 | -- | no |
| `9911753773506421` | no_match | 0.658 | 0.001 | -- | no |
| `994320373506421` | no_match | 0.750 | 0.015 | -- | no |
| `994141873506421` | no_match | 0.825 | 0.049 | -- | no |
| `9912111803506421` | no_match | 0.708 | 0.005 | -- | no |
| `9928969723506421` | no_match | 0.674 | 0.080 | -- | no |
| `9969626263506421` | no_match | 0.669 | 0.041 | -- | yes |
| `9917736573506421` | no_match | 0.690 | 0.006 | -- | yes |
| `992709313506421` | no_match | 0.679 | 0.005 | -- | no |
| `9917172163506421` | match | 0.637 | 0.453 | -- | yes |
| `992445023506421` | no_match | 0.838 | 0.010 | -- | yes |
| `9917170793506421` | no_match | 0.868 | 0.173 | generic_title | yes |
| `9916781353506421` | no_match | 0.744 | 0.367 | generic_title | yes |
| `9928811653506421` | no_match | 0.747 | 0.078 | -- | no |
| `9911809583506421` | no_match | 0.750 | 0.005 | -- | no |
| `9990281473506421` | no_match | 0.940 | 0.019 | -- | yes |
| `9911753913506421` | no_match | 0.740 | 0.009 | generic_title | yes |
| `9927099793506421` | no_match | 0.750 | 0.004 | -- | yes |
| `9926401183506421` | no_match | 0.750 | 0.001 | -- | yes |
| `99125488391606421` | no_match | 0.718 | 0.038 | -- | no |
| `9927553313506421` | no_match | 0.725 | 0.252 | marc_whole_cce_part | yes |
| `9917296603506421` | match | 0.640 | 0.534 | -- | no |
| `9953022333506421` | match | 0.625 | 0.442 | -- | no |
| `9917398923506421` | match | 0.547 | 0.729 | -- | no |
| `9916070663506421` | match | 0.642 | 0.963 | -- | no |

### Weighted-mean right, LightGBM wrong (33 pairs, showing up to 25)

| marc_control_id | truth | weighted | lgbm | categories | note? |
|:---|:---|---:|---:|:---|:---|
| `9985739723506421` | match | 0.702 | 0.211 | -- | no |
| `9911941423506421` | match | 0.791 | 0.237 | -- | no |
| `9920249833506421` | match | 0.723 | 0.277 | -- | no |
| `9925174633506421` | match | 0.727 | 0.169 | -- | no |
| `9917221823506421` | no_match | 0.625 | 0.951 | -- | no |
| `9927596913506421` | no_match | 0.633 | 0.806 | -- | no |
| `998205883506421` | no_match | 0.648 | 0.587 | -- | no |
| `9923856593506421` | no_match | 0.604 | 0.837 | marc_whole_cce_part | yes |
| `994153263506421` | no_match | 0.607 | 0.600 | -- | no |
| `991886873506421` | no_match | 0.625 | 0.442 | -- | no |
| `9929860013506421` | match | 0.655 | 0.262 | -- | no |
| `9930330873506421` | no_match | 0.622 | 0.492 | -- | no |
| `9916767783506421` | no_match | 0.607 | 0.953 | -- | no |
| `9933105333506421` | no_match | 0.605 | 0.527 | -- | no |
| `9964294873506421` | no_match | 0.639 | 0.452 | -- | no |
| `9916332783506421` | no_match | 0.618 | 0.492 | -- | no |
| `9911606923506421` | match | 0.691 | 0.265 | -- | no |
| `9928744783506421` | no_match | 0.625 | 0.632 | -- | no |
| `99125488858406421` | no_match | 0.625 | 0.632 | -- | no |
| `9989203383506421` | no_match | 0.647 | 0.431 | same_title_different_work, generic_title | no |
| `9916779783506421` | no_match | 0.612 | 0.851 | different_edition | no |
| `9917328533506421` | match | 0.657 | 0.199 | -- | no |
| `9912310313506421` | no_match | 0.618 | 0.810 | -- | no |
| `9920947063506421` | no_match | 0.601 | 0.769 | -- | no |
| `99129180505206421` | match | 0.672 | 0.087 | -- | no |

## 8. Decision

Gate inputs:

- Learning curve: **PLATEAU**
- OOF best-F1 beats weighted-mean by ≥ 0.02: **True** (delta +0.0611)
- Full-n fold AUC std ≤ 0.0026 (2026-05-31 run): **False** (std 0.0055)

**HOLD.** The gate is not met. Re-run this diagnostic at the next 500-label increment (~2000 labels) and re-evaluate.
