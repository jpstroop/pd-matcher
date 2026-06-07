# In-pool-but-lost diagnostic — 2026-06-07

**TL;DR**: the #20 recall failure is floor-driven, not ranking-driven — the matcher has the GT at rank 1 in the overwhelming majority of recall losses, but its calibrated score doesn't clear `min_combined_score`. The next instrument should re-examine the `min_combined_score` setting and/or the calibrator fit, not scorer reweighting.

## 1. Experimental setup

Recall investigation over the labeled vault MATCH set. For every MATCH MARC, the production candidate retrieval is run, every retrieved candidate is scored, the full scored list is sorted by `calibrated` descending, and the GT CCE's position is recorded. The diagnostic intentionally bypasses `match_record`'s `qualifying` filter so we can observe what the `min_combined_score` floor would have suppressed.

- **Run date** (UTC): 2026-06-07
- **Vault**: `data/label_vault.jsonl`
- **Candidate pool**: `data/candidates`
- **Index**: `caches/cce.lmdb`
- **MATCH MARCs resolved**: 836
- **MATCH MARCs skipped (not in pool)**: 0
- **MATCH MARCs skipped (GT not in index)**: 0
- **`year_window`**: 0
- **`min_combined_score` floor**: 70.00
- **Floor as calibrated probability**: 0.7000

## 2. Bucket counts

Percentages are of the resolved MATCH MARC corpus. `agree_above_floor` is the matcher returning the right answer; everything else is a recall failure of some kind. `agree_below_floor` is the dominant #20-scope failure: the matcher had the GT at rank 1 but its calibrated score didn't clear `min_combined_score`, so `match_record`'s `qualifying` filter returned `None`. The size of this bucket matches `marcs_evaluated - marcs_with_matcher_top` from `tests/regression/baseline.json`. `out_ranked` + `below_threshold` are the wrong-winner residue; `out_of_pool` is the #19 territory (year-blocked GT) plus genuine retrieval misses.

| bucket | count | % of resolved |
|:---|---:|---:|
| `agree_above_floor` | 706 | 84.45% |
| `agree_below_floor` | 124 | 14.83% |
| `out_ranked` | 6 | 0.72% |
| `below_threshold` | 0 | 0.00% |
| `out_of_pool` | 0 | 0.00% |

### 2a. `out_of_pool` breakdown

`year_blocked`: GT's `reg_year` lies outside the `year_window` of the MARC's `publication_year` (so the year bucket lookup excluded it before any token-set intersection). This is the #19 residue. `missing`: GT shares no title/author/publisher token with the MARC (or has no `reg_year`); it failed token retrieval, which is a different problem.

| sub-reason | count | % of out_of_pool |
|:---|---:|---:|
| `year_blocked` | 0 | 0.00% |
| `missing` | 0 | 0.00% |

## 3. `out_ranked` analysis

Distribution of where the GT lands in the scored list when it's not rank 1, and how big the calibrated-score gap to the winner is.

### 3a. GT rank distribution

| GT rank | count |
|---:|---:|
| 2 | 6 |

### 3b. Score-gap distribution (`winner.calibrated` - `gt.calibrated`)

- **min**: 0.0000  ·  **Q1**: 0.0000  ·  **median**: 0.0000  ·  **Q3**: 0.0000  ·  **max**: 0.0000
- **mean**: 0.0000

### 3c. Per-scorer delta (winner - GT, normalized)

Mean per-scorer `normalized` score of the rank-1 winner minus the GT, averaged across `out_ranked` rows. Positive means the scorer is consistently pushing the wrong winner ahead of the GT. Skipped Evidence contributes a normalized score of `0.0` per the `Evidence.normalized` definition; that's a real signal here (it shows scorer absence as well as scorer disagreement).

| scorer | mean(winner - GT) | n |
|:---|---:|---:|
| `title.token_set` | +0.0000 | 6 |
| `year.delta` | +0.0000 | 6 |
| `edition.compat` | +0.0000 | 6 |
| `isbn.exact` | +0.0000 | 6 |
| `volume.compat` | +0.0000 | 6 |
| `name.author` | -0.1667 | 6 |
| `name.publisher` | -0.1667 | 6 |
| `lccn.exact` | -0.1667 | 6 |
| `extent.page_count` | -0.1667 | 6 |

## 4. Floor-suppressed GT analysis (`agree_below_floor` + `below_threshold`)

GT records that survived candidate retrieval but whose calibrated score fell under the floor and would have been dropped by `match_record`'s `qualifying` filter. Includes both `agree_below_floor` (GT at rank 1) and `below_threshold` (GT not at rank 1).

- **`agree_below_floor`**: 124  ·  **`below_threshold`**: 0

- **GT `calibrated` min**: 0.4059  ·  **Q1**: 0.6344  ·  **median**: 0.6551  ·  **Q3**: 0.6798  ·  **max**: 0.6996
- **mean GT `calibrated`**: 0.6500
- **mean distance below floor** (floor = 0.7000): 0.0500

### 4a. Worst-15 floor-suppressed worktable

Sorted by GT calibrated score ascending; these are the records the floor is most aggressively suppressing.

| rank | pair_id | marc_control_id | nypl_uuid | bucket | gt_rank | gt_score | floor | marc_title | cce_title |
|---:|---:|:---|:---|:---|---:|---:|---:|:---|:---|
| 1 | 553 | `9928849443506421` | `15CFEA96-6F26-1014-8881-C801964016F5` | `agree_below_floor` | 1 | 0.4059 | 0.7000 | Leaves from the Copper Beeches | LEAVES FROM THE COPPER BEECHES. |
| 2 | 556 | `9916838013506421` | `3258ECDC-734A-1014-9154-C7651F0A5C6D` | `agree_below_floor` | 1 | 0.4059 | 0.7000 | "So few" | So few; the immortal record of the Royal air force |
| 3 | 751 | `9916203493506421` | `FC49C6F9-70BA-1014-9292-B7050D9087AA` | `agree_below_floor` | 1 | 0.5062 | 0.7000 | The housing demand of workers in Manhattan | Housing demand of workers in Manhattan |
| 4 | 630 | `9917398923506421` | `7B004E0D-6D03-1014-999C-964E71FFACD8` | `agree_below_floor` | 1 | 0.5471 | 0.7000 | Pier Luigi Nervi: space and structural integrity | Nervi; space and structural integrity. Exhibition, 12 May-1… |
| 5 | 765 | `9912731953506421` | `FBD75DB9-7288-1014-A093-F050B7410E2F` | `agree_below_floor` | 1 | 0.6012 | 0.7000 | La technique | La technique; ou, L'enjeu du siècle. A. Colin. |
| 6 | 683 | `9947727023506421` | `3C5F43B6-734A-1014-8022-9B0F6216A222` | `agree_below_floor` | 1 | 0.6012 | 0.7000 | Scuttlebutt va a la Guerra | Scuttlebutt goes to war |
| 7 | 632 | `9917557413506421` | `731B6597-6C3C-1014-98E6-B36E85AF8D1C` | `agree_below_floor` | 1 | 0.6048 | 0.7000 | The humanities and the curriculum | The humanities and the curriculum; papers from a conference… |
| 8 | 806 | `9925313813506421` | `7813D2BD-7240-1014-AB88-AD98078397AF` | `agree_below_floor` | 1 | 0.6056 | 0.7000 | Défense de la liberté individuelle | Defense de la liberté Individuelle. Paris. |
| 9 | 767 | `9948658983506421` | `17CFA163-72C4-1014-B53A-E905A29103D3` | `agree_below_floor` | 1 | 0.6062 | 0.7000 | Nomades du soleil | Nomades du solell. Lausanne, Guilde du Livre. |
| 10 | 677 | `9912503363506421` | `B1A54A28-6BFB-1014-B6FB-B9486FFAA365` | `agree_below_floor` | 1 | 0.6070 | 0.7000 | Lotte secondarie | Lotte secondarle. |
| 11 | 771 | `9911808803506421` | `0C09C6AD-6F26-1014-8881-C801964016F5` | `agree_below_floor` | 1 | 0.6096 | 0.7000 | El Soldado mexicano, 1837-1847 | El soldado mexicano. The Mexican soldier, 1837-1847. Mexico… |
| 12 | 642 | `9929790083506421` | `676E62D8-7240-1014-AB88-AD98078397AF` | `agree_below_floor` | 1 | 0.6177 | 0.7000 | Sha̓ a-la-ko̓ Mana | Sh'a a-la-k'o mana |
| 13 | 784 | `9924024913506421` | `B76D500E-6F5B-1014-A2B7-83BFD15252F6` | `agree_below_floor` | 1 | 0.6212 | 0.7000 | Les paysages de la mer de la surface a l'abime | Les paysages de la mer; de la surface à l'abîme. 110 photos… |
| 14 | 635 | `9911322493506421` | `27BA5B98-6C48-1014-B56A-C4C8B59DAA18` | `agree_below_floor` | 1 | 0.6213 | 0.7000 | Seminar on complex multiplication | SEMINAR ON COMPLEX MULTIPLICATION; seminar held at the Inst… |
| 15 | 684 | `9927150763506421` | `4A81EC64-71C6-1014-8134-C93186857334` | `agree_below_floor` | 1 | 0.6217 | 0.7000 | Gramática del Quechua ayacuchano | Ayacucho Quechua grammar. |

## 5. `out_ranked` worktable (top-15 by score gap)

Sorted by `winner.calibrated - gt.calibrated` descending. Same shape as the LightGBM disagreement table.

| rank | pair_id | marc_control_id | nypl_uuid | marc_title | cce_title | winner | gt | |delta| |
|---:|---:|:---|:---|:---|:---|---:|---:|---:|
| 1 | 199 | `9916953003506421` | `F8E22B3A-7672-1014-B54D-FD0DCA39AFE7` | The Parthenon of Pericles and its reproduction in America | The Parthenon of Pericles and its reproduction in America | 1.0000 | 1.0000 | 0.0000 |
| 2 | 243 | `9911806703506421` | `1908EDD6-6C3D-1014-98E6-B36E85AF8D1C` | Radioecological concentration processes | Radioecological concentration processes | 1.0000 | 1.0000 | 0.0000 |
| 3 | 320 | `9921010903506421` | `F4A7B2F0-70BE-1014-9473-CECB846257BF` | Mine and countermine | Mine and countermine. | 1.0000 | 1.0000 | 0.0000 |
| 4 | 326 | `9916503123506421` | `EA90D9C5-70BA-1014-A9E8-9AC827975E90` | The flying Dutchman | Flying Dutchman. | 0.8462 | 0.8462 | 0.0000 |
| 5 | 444 | `9911512563506421` | `B83568D1-6F10-1014-90C3-93BD933BF4A5` | Nucleic acid metabolism, cell differentiation, and cancer g… | Nucleic acid metabolism cell differentiation and cancer gro… | 1.0000 | 1.0000 | 0.0000 |
| 6 | 560 | `9916976103506421` | `F9F2A554-72EC-1014-A4F0-E789B14BB857` | Readings in the sociology of migration | Readings in the sociology of migration | 0.7647 | 0.7647 | 0.0000 |
