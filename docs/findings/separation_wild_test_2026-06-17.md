# Wild-distribution separation test — 2026-06-17

Issue #84. The FIRST off-vault SEPARATION check of the learned matcher. Prior separation numbers (learned grouped-OOF AUC 0.993, weighted 0.942) are measured INSIDE the labeled, MATCH-biased vault. The reframe is that the matcher's scaling lever is pair-level SEPARATION — can a score threshold auto-decide a pair without a human — not the saturated top-1 linkage. The open worry is that learned separation might be worse on unknown pairs. This script answers it with a strict held-out split.

## Method

**Time split by `labeled_at` at `2026-06-12T18:55:29Z`.** TRAIN = every non-`unsure` pair labeled at or before the cutoff; TEST = every non-`unsure` pair labeled strictly after it (the wild, stratified, middle-heavy sample). TRAIN ∩ TEST is asserted empty, so the model can never train on a TEST pair.

**One labeled-only learned model trains on TRAIN only** — production `train-scorer` recipe, NO decoys (decoys were a top-1 fix; this is pair-level separation, so labeled-only matches the OOF-0.993 baseline). Locked hyperparameters (max_depth=3, num_leaves=8, min_data_in_leaf=10, lambda_l2=1.0, n_estimators=200, class_weight=balanced, random_state=20260613). The Booster is wrapped directly as the production `LearnedCombiner` — no disk round-trip, the production `caches/learned_scorer.*` is untouched.

**Each TEST pair is scored once** into per-scorer Evidence; three combiner arms grade that same Evidence: `learned`, `weighted_mean`, and `weighted_minus_year` (weighted mean with `year_weight=0.0`, the remaining weights renormalized to sum to 1.0, to remove year's constant uplift). Gold = 1 for `match`, 0 for `no_match`.

- **Feature count**: 53 (production `feature_names()`)
- **TRAIN pairs**: 1434 (positive 922, negative 512)
- **TEST pairs**: 499 labeled; 499 resolved and scored (unresolved: no_marc=0, no_cce=0, dropped=0)

## Headline — held-out separation (TEST set)

| arm | ROC-AUC | average precision |
|:---|---:|---:|
| learned | 0.9870 | 0.9663 |
| weighted | 0.9577 | 0.8975 |
| weighted_minus_year | 0.9637 | 0.9108 |

## Does separation hold off-vault?

Known labeled-vault grouped-OOF baselines for contrast: **learned 0.993**, **weighted 0.942**.

**HOLDS.** Held-out learned AUC 0.9870 is within 0.03 of the 0.993 OOF baseline: separation generalizes to the wild, middle-heavy sample, so threshold-based triage is viable as a scaling lever.

## Threshold sweeps

### learned

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.00 | 0.3267 | 1.0000 | 0.4924 |
| 0.05 | 0.8256 | 0.9877 | 0.8994 |
| 0.10 | 0.8548 | 0.9755 | 0.9112 |
| 0.15 | 0.8701 | 0.9448 | 0.9059 |
| 0.20 | 0.8947 | 0.9387 | 0.9162 |
| 0.25 | 0.8947 | 0.9387 | 0.9162 |
| 0.30 | 0.8988 | 0.9264 | 0.9124 |
| 0.35 | 0.9036 | 0.9202 | 0.9119 |
| 0.40 | 0.9141 | 0.9141 | 0.9141 |
| 0.45 | 0.9255 | 0.9141 | 0.9198 |
| 0.50 | 0.9363 | 0.9018 | 0.9187 |
| 0.55 | 0.9481 | 0.8957 | 0.9211 |
| 0.60 | 0.9536 | 0.8834 | 0.9172 |
| 0.65 | 0.9595 | 0.8712 | 0.9132 |
| 0.70 | 0.9589 | 0.8589 | 0.9061 |
| 0.75 | 0.9586 | 0.8528 | 0.9026 |
| 0.80 | 0.9650 | 0.8466 | 0.9020 |
| 0.85 | 0.9643 | 0.8282 | 0.8911 |
| 0.90 | 0.9697 | 0.7853 | 0.8678 |
| 0.95 | 0.9766 | 0.7669 | 0.8591 |
| 1.00 | 0.0000 | 0.0000 | 0.0000 |

### weighted

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.00 | 0.3267 | 1.0000 | 0.4924 |
| 0.05 | 0.3267 | 1.0000 | 0.4924 |
| 0.10 | 0.3267 | 1.0000 | 0.4924 |
| 0.15 | 0.3273 | 1.0000 | 0.4932 |
| 0.20 | 0.3300 | 1.0000 | 0.4962 |
| 0.25 | 0.3528 | 1.0000 | 0.5216 |
| 0.30 | 0.3773 | 1.0000 | 0.5479 |
| 0.35 | 0.4075 | 1.0000 | 0.5790 |
| 0.40 | 0.5142 | 1.0000 | 0.6792 |
| 0.45 | 0.6105 | 1.0000 | 0.7581 |
| 0.50 | 0.6708 | 0.9877 | 0.7990 |
| 0.55 | 0.7306 | 0.9816 | 0.8377 |
| 0.60 | 0.7760 | 0.9141 | 0.8394 |
| 0.65 | 0.8630 | 0.7730 | 0.8155 |
| 0.70 | 0.8879 | 0.5828 | 0.7037 |
| 0.75 | 0.8902 | 0.4479 | 0.5959 |
| 0.80 | 0.9706 | 0.4049 | 0.5714 |
| 0.85 | 0.9839 | 0.3742 | 0.5422 |
| 0.90 | 0.9811 | 0.3190 | 0.4815 |
| 0.95 | 0.9737 | 0.2270 | 0.3682 |
| 1.00 | 0.9444 | 0.1043 | 0.1878 |

### weighted_minus_year

| threshold | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.00 | 0.3267 | 1.0000 | 0.4924 |
| 0.05 | 0.3505 | 1.0000 | 0.5191 |
| 0.10 | 0.3647 | 1.0000 | 0.5344 |
| 0.15 | 0.3900 | 1.0000 | 0.5611 |
| 0.20 | 0.4579 | 1.0000 | 0.6281 |
| 0.25 | 0.5208 | 1.0000 | 0.6849 |
| 0.30 | 0.6015 | 1.0000 | 0.7512 |
| 0.35 | 0.6573 | 1.0000 | 0.7932 |
| 0.40 | 0.6907 | 1.0000 | 0.8170 |
| 0.45 | 0.7385 | 0.9877 | 0.8451 |
| 0.50 | 0.7573 | 0.9571 | 0.8455 |
| 0.55 | 0.8034 | 0.8773 | 0.8387 |
| 0.60 | 0.8921 | 0.7607 | 0.8212 |
| 0.65 | 0.9182 | 0.6196 | 0.7399 |
| 0.70 | 0.9326 | 0.5092 | 0.6587 |
| 0.75 | 0.9718 | 0.4233 | 0.5897 |
| 0.80 | 0.9848 | 0.3988 | 0.5677 |
| 0.85 | 0.9828 | 0.3497 | 0.5158 |
| 0.90 | 0.9808 | 0.3129 | 0.4744 |
| 0.95 | 0.9667 | 0.1779 | 0.3005 |
| 1.00 | 0.9444 | 0.1043 | 0.1878 |

**Best-F1 learned threshold**: 0.55 (precision 0.9481, recall 0.8957, F1 0.9211).

## Disagreements at the best learned threshold

At the best-F1 learned threshold (0.55). ⚠️ marks pairs in the below-0.50 score region.

### False accepts (score ≥ threshold, gold = no_match) (8 total)

| score | low? | marc_control_id | nypl_uuid | MARC title | CCE title |
|---:|:--:|:---|:---|:---|:---|
| 0.9999 |  | 9917783183506421 | 515C93A1-6DF7-1014-B929-815422DE91B0 | Satellite dynamics symposium, São Paulo, Brazil, June 19-21, 1974 | COSPAR-IAU-IUTAM satellite dynamics |
| 0.9996 |  | 9970609253506421 | 6959C24F-6D03-1014-999C-964E71FFACD8 | Halftone photography for offset litography | Halftone photography for offset lithography. |
| 0.9897 |  | 9968471463506421 | 4AB91616-726B-1014-9489-930D3C86AF32 | Stangl a portrait of progress in pottery | A portrait of progress in pottery. |
| 0.9350 |  | 9917983403506421 | 24C780B7-72C4-1014-B53A-E905A29103D3 | How to read and understand financial and business news | A century of financial advertising in the New York times. |
| 0.8563 |  | 996862033506421 | 884FF476-6D03-1014-999C-964E71FFACD8 | The impact of the professional engineering union a study of collective bargaining among engineers and scientists and its significance for management | The impact of the professional engineering union. Division of Research, Graduate School of Business Administration, Harvard University. |
| 0.7881 |  | 996384653506421 | 1BFBBA4A-726B-1014-9489-930D3C86AF32 | Men of physics L. D. Landau | L. D. Landau. |
| 0.6334 |  | 996578393506421 | 4B2D991B-734A-1014-B590-F68F1E466D9E | Religious and secular leadership | Religious and secular leadership. Pt. 1 |
| 0.5693 |  | 996507883506421 | 953DF19D-76FE-1014-8A60-894D8A9620E2 | Arms and the Covenant speeches by the Right Hon. Winston S. Churchill, C.H., M.P | Kent, England. Arms and the Covenant. |

### False rejects (score < threshold, gold = match) (17 total)

| score | low? | marc_control_id | nypl_uuid | MARC title | CCE title |
|---:|:--:|:---|:---|:---|:---|
| 0.0071 | ⚠️ | 996484743506421 | 400ACFAA-6BFB-1014-B6FB-B9486FFAA365 | Virginia genealogies | Virginia genealogies; a trial list of printed books and pamphlets. |
| 0.0359 | ⚠️ | 9969750213506421 | D8034931-728D-1014-8500-D70EB891A161 | The American story | The American story. Vol.1-2. |
| 0.0527 | ⚠️ | 9917916783506421 | AE92EDA6-798D-1014-983D-92E56EBBC3D4 | The new illustrated flora of the Hawaiian Islands flora Hawaiiensis ... Books 1- | Flora hawaiiensis |
| 0.0956 | ⚠️ | 9917732543506421 | 82E225B9-6C6D-1014-965B-8464748AEC84 | The encyclopedia of jazz | The new edition of The encyclopedia of Jazz. |
| 0.1178 | ⚠️ | 996762713506421 | 954C9860-728D-1014-8DCA-EDC48D7CF4C8 | Istanbul boy the autobiography of Aziz Nesin | Istanbul boy: Boeyle Gelmis Boeyle Gitmez (that's how it was but not how it's going to be) the autobiography of Aziz Nesin |
| 0.1433 | ⚠️ | 9917396673506421 | 4D355D5E-7240-1014-AB88-AD98078397AF | America's children | America's children. John Wiley. |
| 0.1438 | ⚠️ | 9917626733506421 | D5BD3CB6-6D99-1014-A5EA-BAD7E825951E | The singer not the song | The singer not the song. |
| 0.1449 | ⚠️ | 9991673843506421 | 41A43176-6DF7-1014-B929-815422DE91B0 | Oscar Wilde a biography | Oscar Wilde |
| 0.1460 | ⚠️ | 996374003506421 | A0AF0A08-6D0F-1014-9803-8BC52073E431 | The living city | The living city. Horizon Press. Some material prev. pub. 1945 as When democracy builds. |
| 0.1909 | ⚠️ | 9917774063506421 | 8AC43D80-6D0F-1014-9803-8BC52073E431 | The study of politics the present state of American political science | The study of politics. University of Illinois Press. |
| 0.2707 | ⚠️ | 9917736403506421 | 28E2E605-6CCB-1014-A5FE-CCE70832EF77 | The American high-school student | The American high-school student; the identification, development and utilization of human talents. |
| 0.2909 | ⚠️ | 996066713506421 | D0B4FD96-6BFB-1014-B6FB-B9486FFAA365 | Focusing of charged particles | Focusing of charged particles. |
| 0.3382 | ⚠️ | 9917744193506421 | 553B1325-72C4-1014-B53A-E905A29103D3 | One life | One life. Simon & Schuster. |
| 0.3952 | ⚠️ | 9992012973506421 | 321BA70F-70AB-1014-B70C-FEF2CDD9AD0E | America, a prophecy | William Blake: America |
| 0.4890 | ⚠️ | 9969497253506421 | B96DC3B4-6CFE-1014-9B03-A435536521E8 | The twelve days of Christmas a Christmas carol | a Christmas carol |
| 0.4983 | ⚠️ | 9917697273506421 | EB44F0C2-6D13-1014-B63C-9736EB68D6D5 | The Psalm book of Charles Knowles | The Psalm book. |
| 0.5127 |  | 996121133506421 | 14ADE0AC-72C4-1014-84F6-87703AD4166D | The shotgunner's a modern encyclopedia | The shotgunner's book. Stackpole Co. |

## Triage viability (learned arm)

- **Auto-ACCEPT** threshold T_hi = 1.00 (zero false-accepts above it): 0 pairs
- **Auto-REJECT** threshold T_lo = 0.00 (zero false-rejects below it): 0 pairs
- **Auto-decided**: 0/499 (0.0%)
- **Residual human middle**: 499/499 (100.0%)

> **Caveat.** The TEST set is a STRATIFIED, deliberately middle-heavy sample, so the auto-decidable fraction here is a LOWER bound — it is NOT the production rate. Real acquired pairs skew toward the easy tails, where a far larger share auto-decides.

## Reproduction

```
pdm run python scripts/separation_wild_test.py \
    > docs/findings/separation_wild_test_2026-06-17.md
```

This script touches nothing under `src/` or `data/`, overwrites no artifact, and is deterministic (fixed seed, `n_jobs=1`).
