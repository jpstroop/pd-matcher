eval.vault.marc_not_in_pool marc_control_id=99131652035506421 nypl_uuid=19568F99-6DC2-1014-AB63-A583E9D2BC9B
eval.vault.marc_not_in_pool marc_control_id=99125228572706421 nypl_uuid=7B3EA0EC-6DF3-1014-98F6-A54A55E2F41C
eval.vault.marc_not_in_pool marc_control_id=99131633047806421 nypl_uuid=4A6362AB-6CFA-1014-B19C-B29A16CE9672
eval.vault.marc_not_in_pool marc_control_id=99131652134906421 nypl_uuid=101EEFD3-6D7D-1014-B4A5-D7A31D17AB40
eval.vault.marc_not_in_pool marc_control_id=99131652015506421 nypl_uuid=F703AFBC-6DF2-1014-98F6-A54A55E2F41C
eval.vault.marc_not_in_pool marc_control_id=9995780703506421 nypl_uuid=C56DA2E5-7893-1014-8223-95F882E026CF
eval.vault.marc_not_in_pool marc_control_id=9990543393506421 nypl_uuid=0CEE8AFF-79D5-1014-8221-AE20959418B2
eval.vault.marc_not_in_pool marc_control_id=99131651993606421 nypl_uuid=7EA1F0E4-7222-1014-A92F-CB0463F9F54D
eval.vault.marc_not_in_pool marc_control_id=9983136793506421 nypl_uuid=A075974E-75F5-1014-96EA-B4CA74C85597
/Users/jstroop/workspace/public_domain/.venv/lib/python3.14/site-packages/sklearn/utils/validation.py:2691: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(
/Users/jstroop/workspace/public_domain/.venv/lib/python3.14/site-packages/sklearn/utils/validation.py:2691: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(
/Users/jstroop/workspace/public_domain/.venv/lib/python3.14/site-packages/sklearn/utils/validation.py:2691: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(
/Users/jstroop/workspace/public_domain/.venv/lib/python3.14/site-packages/sklearn/utils/validation.py:2691: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(
/Users/jstroop/workspace/public_domain/.venv/lib/python3.14/site-packages/sklearn/utils/validation.py:2691: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(
<!-- fold 1: roc_auc=0.9843 pr_auc=0.9949 -->
<!-- fold 2: roc_auc=0.9843 pr_auc=0.9951 -->
<!-- fold 3: roc_auc=0.9893 pr_auc=0.9967 -->
<!-- fold 4: roc_auc=0.9929 pr_auc=0.9979 -->
<!-- fold 5: roc_auc=0.9916 pr_auc=0.9974 -->
# Learned-scorer diagnostic — 2026-05-29

## 1. Experimental setup

- **Pairs scored**: 948 (735 match / 213 no_match)
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

- **ROC-AUC** across folds: mean=0.9885 std=0.0036
- **PR-AUC** across folds: mean=0.9964 std=0.0012

> The negative class is the binding constraint at this corpus size; fold-level numbers are directional, not deployable. Recompute at ~1500 labels.

## 2. Feature importance ranking

LightGBM `gain` importance averaged across the five folds, normalized to sum to 1.0 across all features. The `current_weight` column shows the matching.yaml weight for that scorer (skipped-flag features have no direct weight analogue and are marked `--`).

| rank | feature | lgbm_importance | current_weight |
|---:|:---|---:|---:|
| 1 | `name.publisher` | 0.3706 | 0.100 |
| 2 | `title.token_set` | 0.2560 | 0.350 |
| 3 | `name.author` | 0.2282 | 0.200 |
| 4 | `extent.page_count` | 0.0824 | 0.050 |
| 5 | `volume.compat__skipped` | 0.0402 | -- |
| 6 | `extent.page_count__skipped` | 0.0181 | -- |
| 7 | `edition.compat` | 0.0025 | 0.050 |
| 8 | `edition.compat__skipped` | 0.0009 | -- |
| 9 | `volume.compat` | 0.0007 | 0.050 |
| 10 | `name.author__skipped` | 0.0002 | -- |
| 11 | `lccn.exact` | 0.0001 | 0.100 |
| 12 | `name.publisher__skipped` | 0.0000 | -- |
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
| `extent.page_count` | 1.5926 | 1.6273 | bidirectional |
| `title.token_set` | 1.5908 | 2.0962 | bidirectional |
| `name.publisher` | 1.4154 | 1.6387 | bidirectional |
| `name.author` | 1.2316 | 1.4072 | bidirectional |
| `extent.page_count__skipped` | 0.3469 | 0.3678 | bidirectional |
| `volume.compat__skipped` | 0.1790 | 0.5290 | positive |
| `edition.compat` | 0.0775 | 0.1582 | negative |
| `edition.compat__skipped` | 0.0289 | 0.1166 | bidirectional |
| `name.author__skipped` | 0.0065 | 0.0237 | negative |
| `volume.compat` | 0.0046 | 0.0193 | bidirectional |
| `lccn.exact` | 0.0037 | 0.0119 | negative |
| `name.publisher__skipped` | 0.0005 | 0.0023 | bidirectional |
| `isbn.exact` | 0.0000 | 0.0000 | inert |
| `year.delta` | 0.0000 | 0.0000 | inert |
| `title.token_set__skipped` | 0.0000 | 0.0000 | inert |
| `year.delta__skipped` | 0.0000 | 0.0000 | inert |
| `lccn.exact__skipped` | 0.0000 | 0.0000 | inert |
| `isbn.exact__skipped` | 0.0000 | 0.0000 | inert |

## 4. Top-30 disagreement pairs

Pairs sorted by `|lgbm_pred - combined_score|` (descending). `verdict` is the human label. Use this as the worktable for the next round of weight inspection.

| rank | pair_id | marc_control_id | nypl_uuid | verdict | combined | lgbm | |delta| | marc_title | cce_title |
|---:|---:|:---|:---|:---|---:|---:|---:|:---|:---|
| 1 | 634 | `9990281473506421` | `5E1E18EC-7950-1014-8674-C1ED2059688C` | no_match | 0.940 | 0.039 | 0.900 | The Tudors | A |
| 2 | 613 | `992445023506421` | `AFF5E6E9-72DD-1014-B3A3-DDF21CF07E3D` | no_match | 0.838 | 0.012 | 0.826 | Anglo-Norman armory | ? |
| 3 | 269 | `994141873506421` | `D851443E-6C47-1014-802D-A968AD0E9A77` | no_match | 0.825 | 0.035 | 0.790 | Untitled 2 & 3 | Untitled |
| 4 | 584 | `995086263506421` | `AAF9D6C0-7673-1014-B817-D52E0321EE79` | no_match | 0.774 | 0.001 | 0.773 | Jewish maritime revival | סִפּוּרִים מֵאֵת יׅצְחׇק לֵיבּוּשׁ פֶּרֶץ. |
| 5 | 946 | `9923621713506421` | `00351C7A-6D7D-1014-B4A5-D7A31D17AB40` | no_match | 0.774 | 0.005 | 0.768 | Éditions originales de romantiques et d'auteurs contemporai… | פאַרקאך־בוך געזונטהייט פון לינע בו־אַון. |
| 6 | 507 | `9928868433506421` | `AAEC67BC-7673-1014-B817-D52E0321EE79` | no_match | 0.763 | 0.001 | 0.762 | [Paul Tulane | בלאנדזשענדע מאָטיווען. |
| 7 | 617 | `9917170793506421` | `3CC51FE1-70AF-1014-98B5-BD1CA90B7FB2` | no_match | 0.799 | 0.040 | 0.760 | Report and selected papers | Selecting annual report papers. |
| 8 | 930 | `9926895063506421` | `AAE4BDB3-7673-1014-B817-D52E0321EE79` | no_match | 0.756 | 0.002 | 0.755 | The geology of the townships of Gaboury and Blondeau, Temis… | הכנסת כלה. כרד ראשון ושני. |
| 9 | 666 | `9951809883506421` | `30443B6C-734A-1014-9AF0-99117F0389BD` | no_match | 0.750 | 0.001 | 0.749 | [Pétainist alphabet blocks] | די מפרשים פון דער תּורה |
| 10 | 622 | `9911809583506421` | `0F3DC319-73D0-1014-A526-C29EA9D2D19F` | no_match | 0.750 | 0.001 | 0.749 | Graduation exercises, Bolinas School | The |
| 11 | 576 | `9911864563506421` | `30443B6C-734A-1014-9AF0-99117F0389BD` | no_match | 0.750 | 0.001 | 0.749 | A Showing of the private collection of Mr. & Mrs. Frederic … | די מפרשים פון דער תּורה |
| 12 | 661 | `9926401183506421` | `0F3DC319-73D0-1014-A526-C29EA9D2D19F` | no_match | 0.750 | 0.002 | 0.748 | A Style of our own | The |
| 13 | 894 | `9929881443506421` | `F589566E-7454-1014-BDF7-E78324322810` | no_match | 0.750 | 0.002 | 0.748 | American business polices | This is it |
| 14 | 920 | `9930115393506421` | `28CC8BDE-72D4-1014-932B-E4B92D1A29C1` | no_match | 0.747 | 0.002 | 0.745 | The development of Russian commerce on the Black Sea and it… | פון אייביגען קוואל. געדאנקען און מיינונגען פון דער "אגדה" א… |
| 15 | 922 | `9926941623506421` | `AAEC67BC-7673-1014-B817-D52E0321EE79` | no_match | 0.743 | 0.001 | 0.742 | Glycofuranosides and thioglycofuranosides | בלאנדזשענדע מאָטיווען. |
| 16 | 84 | `9917021493506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.010 | 0.740 | Victorian photographs of famous men & fair women | Where to? |
| 17 | 638 | `9911753913506421` | `811C1A97-6CA4-1014-AFF1-C0B6E171A089` | no_match | 0.740 | 0.006 | 0.735 | Proceedings | Proceedings. |
| 18 | 909 | `9916882783506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.015 | 0.735 | Glossary of geology | Where to? |
| 19 | 648 | `9927099793506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.016 | 0.734 | Friday Mount: first settlement at Holbrook and the south-we… | Where to? |
| 20 | 898 | `9932017723506421` | `4A4D1F73-6D06-1014-A65A-D57CF5CB45BA` | match | 0.768 | 0.036 | 0.732 | Now that April's there | Now that April's there |
| 21 | 927 | `9917014243506421` | `0F3DC319-73D0-1014-A526-C29EA9D2D19F` | no_match | 0.733 | 0.002 | 0.732 | Juvenal and declamtion | The |
| 22 | 121 | `994320373506421` | `07EBCABD-6D2E-1014-A550-82A757018889` | no_match | 0.750 | 0.021 | 0.729 | Chaucer's Constance and accused queens | Where to? |
| 23 | 591 | `9911690183506421` | `0286F20D-7455-1014-98F9-DB39C8E8A4AD` | match | 0.768 | 0.041 | 0.727 | Essays in biology in honor of Herbert M. Evans | Essays in biology in honor of Herbert M. Evans, written by … |
| 24 | 889 | `9929729623506421` | `513A7200-7119-1014-B0E4-A14A79DE7F92` | no_match | 0.716 | 0.002 | 0.714 | The dead star | Who is he? |
| 25 | 932 | `9929234863506421` | `95436E7F-76FE-1014-8A60-894D8A9620E2` | no_match | 0.716 | 0.001 | 0.714 | The Cuban question as reflected in the editorial columns of… | Гибель Атлантинова. Съ иллюстрапіями А. Н. Авинова. |
| 26 | 621 | `9928811653506421` | `72A44A91-6D0F-1014-9803-8BC52073E431` | no_match | 0.747 | 0.033 | 0.714 | Selected poems | A selection of poems. |
| 27 | 918 | `9915951973506421` | `DB7BFC82-728D-1014-8500-D70EB891A161` | no_match | 0.710 | 0.002 | 0.707 | Collected poems | Collected poems. |
| 28 | 917 | `9933244693506421` | `513A7200-7119-1014-B0E4-A14A79DE7F92` | no_match | 0.707 | 0.002 | 0.705 | The Sacco-Vanzetti case | Who is he? |
| 29 | 300 | `9912111803506421` | `A870FAEB-7E9A-1014-A737-EA626E1B7465` | no_match | 0.708 | 0.005 | 0.704 | Cold war to détente | From the cold war to detente |
| 30 | 933 | `9926930823506421` | `038C305C-70BF-1014-9AFF-F4977B7448FF` | no_match | 0.701 | 0.001 | 0.700 | Locally connected spaces and generalized manifolds | רש׳י-דער פאׇלקם־לערעד; פון מנהם ג. גלען. |
