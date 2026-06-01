<!-- fold 1: roc_auc=0.9924 pr_auc=0.9971 -->
<!-- fold 2: roc_auc=0.9945 pr_auc=0.9980 -->
<!-- fold 3: roc_auc=0.9876 pr_auc=0.9955 -->
<!-- fold 4: roc_auc=0.9891 pr_auc=0.9963 -->
<!-- fold 5: roc_auc=0.9933 pr_auc=0.9976 -->
# Learned-scorer diagnostic — 2026-05-29

## 1. Experimental setup

- **Pairs scored**: 1039 (770 match / 269 no_match)
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

- **ROC-AUC** across folds: mean=0.9914 std=0.0026
- **PR-AUC** across folds: mean=0.9969 std=0.0009

> The negative class is the binding constraint at this corpus size; fold-level numbers are directional, not deployable. Recompute at ~1500 labels.

## 2. Feature importance ranking

LightGBM `gain` importance averaged across the five folds, normalized to sum to 1.0 across all features. The `current_weight` column shows the matching.yaml weight for that scorer (skipped-flag features have no direct weight analogue and are marked `--`).

| rank | feature | lgbm_importance | current_weight |
|---:|:---|---:|---:|
| 1 | `name.author` | 0.2908 | 0.200 |
| 2 | `title.token_set` | 0.2834 | 0.350 |
| 3 | `name.publisher` | 0.2411 | 0.100 |
| 4 | `extent.page_count` | 0.1210 | 0.050 |
| 5 | `volume.compat__skipped` | 0.0339 | -- |
| 6 | `extent.page_count__skipped` | 0.0253 | -- |
| 7 | `edition.compat` | 0.0026 | 0.050 |
| 8 | `edition.compat__skipped` | 0.0007 | -- |
| 9 | `name.publisher__skipped` | 0.0004 | -- |
| 10 | `name.author__skipped` | 0.0004 | -- |
| 11 | `volume.compat` | 0.0003 | 0.050 |
| 12 | `lccn.exact` | 0.0000 | 0.100 |
| 13 | `isbn.exact` | 0.0000 | 0.000 |
| 14 | `year.delta` | 0.0000 | 0.100 |
| 15 | `title.token_set__skipped` | 0.0000 | -- |
| 16 | `year.delta__skipped` | 0.0000 | -- |
| 17 | `lccn.exact__skipped` | 0.0000 | -- |
| 18 | `isbn.exact__skipped` | 0.0000 | -- |

## 3. Per-feature SHAP contribution distributions

Mean absolute contribution, standard deviation, and a coarse direction label across all out-of-fold predictions. High `std` relative to `mean_abs` signals interaction effects: the feature pushes predictions in different directions in different contexts.

| feature | mean_abs | std | direction |
|:---|---:|---:|:---|
| `extent.page_count` | 1.8330 | 1.8696 | bidirectional |
| `title.token_set` | 1.6882 | 2.2133 | bidirectional |
| `name.author` | 1.4754 | 1.6532 | bidirectional |
| `name.publisher` | 1.3500 | 1.5346 | bidirectional |
| `extent.page_count__skipped` | 0.4210 | 0.4457 | bidirectional |
| `volume.compat__skipped` | 0.1616 | 0.4844 | positive |
| `edition.compat` | 0.0729 | 0.1585 | negative |
| `edition.compat__skipped` | 0.0263 | 0.1151 | negative |
| `name.author__skipped` | 0.0148 | 0.0368 | negative |
| `name.publisher__skipped` | 0.0088 | 0.0282 | negative |
| `volume.compat` | 0.0028 | 0.0119 | bidirectional |
| `isbn.exact` | 0.0000 | 0.0000 | inert |
| `year.delta` | 0.0000 | 0.0000 | inert |
| `title.token_set__skipped` | 0.0000 | 0.0000 | inert |
| `year.delta__skipped` | 0.0000 | 0.0000 | inert |
| `lccn.exact__skipped` | 0.0000 | 0.0000 | inert |
| `isbn.exact__skipped` | 0.0000 | 0.0000 | inert |
| `lccn.exact` | 0.0000 | 0.0000 | inert |

## 4. Top-30 disagreement pairs

Pairs sorted by `|lgbm_pred - combined_score|` (descending). `verdict` is the human label. Use this as the worktable for the next round of weight inspection.

| rank | pair_id | marc_control_id | nypl_uuid | verdict | combined | lgbm | |delta| | marc_title | cce_title |
|---:|---:|:---|:---|:---|---:|---:|---:|:---|:---|
| 1 | 632 | `9990281473506421` | `5E1E18EC-7950-1014-8674-C1ED2059688C` | no_match | 0.940 | 0.045 | 0.894 | The Tudors | A |
| 2 | 612 | `992445023506421` | `AFF5E6E9-72DD-1014-B3A3-DDF21CF07E3D` | no_match | 0.838 | 0.018 | 0.820 | Anglo-Norman armory | ? |
| 3 | 180 | `9929739923506421` | `C5AFA207-6F5B-1014-A2B7-83BFD15252F6` | match | 0.832 | 0.039 | 0.793 | Don Juan de Oñate, colonizer of New Mexico, 1595-1628 | Don Juan de Offate, colonizer of New Mexico, 1595-1628 |
| 4 | 616 | `9917170793506421` | `3CC51FE1-70AF-1014-98B5-BD1CA90B7FB2` | no_match | 0.799 | 0.048 | 0.751 | Report and selected papers | Selecting annual report papers. |
| 5 | 621 | `9911809583506421` | `0F3DC319-73D0-1014-A526-C29EA9D2D19F` | no_match | 0.750 | 0.001 | 0.749 | Graduation exercises, Bolinas School | The |
| 6 | 659 | `9926401183506421` | `0F3DC319-73D0-1014-A526-C29EA9D2D19F` | no_match | 0.750 | 0.001 | 0.749 | A Style of our own | The |
| 7 | 893 | `9929881443506421` | `F589566E-7454-1014-BDF7-E78324322810` | no_match | 0.750 | 0.001 | 0.749 | American business polices | This is it |
| 8 | 267 | `994141873506421` | `D851443E-6C47-1014-802D-A968AD0E9A77` | no_match | 0.825 | 0.079 | 0.746 | Untitled 2 & 3 | Untitled |
| 9 | 646 | `9927099793506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.008 | 0.742 | Friday Mount: first settlement at Holbrook and the south-we… | Where to? |
| 10 | 83 | `9917021493506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.009 | 0.741 | Victorian photographs of famous men & fair women | Where to? |
| 11 | 120 | `994320373506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.009 | 0.741 | Chaucer's Constance and accused queens | Where to? |
| 12 | 636 | `9911753913506421` | `811C1A97-6CA4-1014-AFF1-C0B6E171A089` | no_match | 0.740 | 0.004 | 0.737 | Proceedings | Proceedings. |
| 13 | 908 | `9916882783506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.014 | 0.736 | Glossary of geology | Where to? |
| 14 | 926 | `9917014243506421` | `0F3DC319-73D0-1014-A526-C29EA9D2D19F` | no_match | 0.733 | 0.001 | 0.733 | Juvenal and declamtion | The |
| 15 | 684 | `99125488391606421` | `953AB397-7176-1014-A92C-DCC6D41F8A68` | no_match | 0.718 | 0.003 | 0.714 | Textures | Textures & designs |
| 16 | 888 | `9929729623506421` | `513A7200-7119-1014-B0E4-A14A79DE7F92` | no_match | 0.716 | 0.002 | 0.714 | The dead star | Who is he? |
| 17 | 620 | `9928811653506421` | `72A44A91-6D0F-1014-9803-8BC52073E431` | no_match | 0.747 | 0.038 | 0.709 | Selected poems | A selection of poems. |
| 18 | 917 | `9915951973506421` | `DB7BFC82-728D-1014-8500-D70EB891A161` | no_match | 0.710 | 0.003 | 0.707 | Collected poems | Collected poems. |
| 19 | 916 | `9933244693506421` | `513A7200-7119-1014-B0E4-A14A79DE7F92` | no_match | 0.707 | 0.002 | 0.704 | The Sacco-Vanzetti case | Who is he? |
| 20 | 298 | `9912111803506421` | `A870FAEB-7E9A-1014-A737-EA626E1B7465` | no_match | 0.708 | 0.005 | 0.704 | Cold war to détente | From the cold war to detente |
| 21 | 896 | `9916748343506421` | `0F3DC319-73D0-1014-A526-C29EA9D2D19F` | no_match | 0.700 | 0.001 | 0.699 | History and cultivation of cotton and tobacco | The |
| 22 | 1035 | `9949672673506421` | `642EB8A5-75F5-1014-96EA-B4CA74C85597` | no_match | 0.692 | 0.001 | 0.691 | El libro infantil de las Fieras | O! |
| 23 | 999 | `9916742333506421` | `761714C2-6C3D-1014-98E6-B36E85AF8D1C` | no_match | 0.689 | 0.001 | 0.688 | Bibliography on clean rooms | Why? |
| 24 | 982 | `9929742253506421` | `1E3F507C-6C3D-1014-98E6-B36E85AF8D1C` | no_match | 0.688 | 0.001 | 0.686 | The nationality of ships | The more. |
| 25 | 415 | `9917736573506421` | `ABB3DBAA-71C5-1014-8134-C93186857334` | no_match | 0.690 | 0.004 | 0.686 | High-energy astrophysics | Proceedings. |
| 26 | 973 | `992756563506421` | `B32D57BD-7950-1014-8674-C1ED2059688C` | no_match | 0.686 | 0.001 | 0.686 | 1776 | The will |
| 27 | 820 | `9948224133506421` | `0CDBDD53-79D5-1014-8221-AE20959418B2` | no_match | 0.687 | 0.002 | 0.685 | A selection of poems | Selected poems |
| 28 | 817 | `9912115073506421` | `3D1CE04B-70AB-1014-B70C-FEF2CDD9AD0E` | no_match | 0.686 | 0.001 | 0.685 | The propaganda of Adolf Hitler | Who is who |
| 29 | 175 | `992323233506421` | `B350A3F6-7176-1014-A92C-DCC6D41F8A68` | no_match | 0.684 | 0.004 | 0.680 | I ching; the book of changes | I Ching: book of change |
| 30 | 592 | `9911706753506421` | `8F8E6CEB-6CC9-1014-9A69-A78FEFE32E94` | no_match | 0.682 | 0.004 | 0.679 | Acquisitions, 1953-62 | Catalogue. |
