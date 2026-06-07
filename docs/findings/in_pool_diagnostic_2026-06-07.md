# In-pool-but-lost diagnostic — 2026-06-07

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

Percentages are of the resolved MATCH MARC corpus. `out_of_pool` is the #19 territory (year-blocked GT) plus genuine retrieval misses; `out_ranked` + `below_threshold` are the #20 in-pool-but-lost scope.

| bucket | count | % of resolved |
|:---|---:|---:|
| `agree` | 832 | 99.52% |
| `out_ranked` | 4 | 0.48% |
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
| 2 | 4 |

### 3b. Score-gap distribution (`winner.calibrated` - `gt.calibrated`)

- **min**: 0.0000  ·  **Q1**: 0.0000  ·  **median**: 0.0000  ·  **Q3**: 0.0000  ·  **max**: 0.0000
- **mean**: 0.0000

### 3c. Per-scorer delta (winner - GT, normalized)

Mean per-scorer `normalized` score of the rank-1 winner minus the GT, averaged across `out_ranked` rows. Positive means the scorer is consistently pushing the wrong winner ahead of the GT. Skipped Evidence contributes a normalized score of `0.0` per the `Evidence.normalized` definition; that's a real signal here (it shows scorer absence as well as scorer disagreement).

| scorer | mean(winner - GT) | n |
|:---|---:|---:|
| `title.token_set` | +0.0000 | 4 |
| `name.author` | +0.0000 | 4 |
| `name.publisher` | +0.0000 | 4 |
| `year.delta` | +0.0000 | 4 |
| `edition.compat` | +0.0000 | 4 |
| `lccn.exact` | +0.0000 | 4 |
| `isbn.exact` | +0.0000 | 4 |
| `extent.page_count` | +0.0000 | 4 |
| `volume.compat` | +0.0000 | 4 |

## 4. `below_threshold` analysis

GT records that survived candidate retrieval but whose calibrated score fell under the floor and would have been dropped by `match_record`'s `qualifying` filter.

_No `below_threshold` rows; nothing to analyze._

## 5. `out_ranked` worktable (top-15 by score gap)

Sorted by `winner.calibrated - gt.calibrated` descending. Same shape as the LightGBM disagreement table.

| rank | pair_id | marc_control_id | nypl_uuid | marc_title | cce_title | winner | gt | |delta| |
|---:|---:|:---|:---|:---|:---|---:|---:|---:|
| 1 | 101 | `99108130433506421` | `193C749F-6CC9-1014-9A69-A78FEFE32E94` | The material relics of music in ancient Palestine and its e… | The material relics of music in ancient Palestine and its e… | 0.9700 | 0.9700 | 0.0000 |
| 2 | 243 | `9911806703506421` | `1908EDD6-6C3D-1014-98E6-B36E85AF8D1C` | Radioecological concentration processes | Radioecological concentration processes | 1.0000 | 1.0000 | 0.0000 |
| 3 | 326 | `9916503123506421` | `EA90D9C5-70BA-1014-A9E8-9AC827975E90` | The flying Dutchman | Flying Dutchman. | 0.8462 | 0.8462 | 0.0000 |
| 4 | 560 | `9916976103506421` | `F9F2A554-72EC-1014-A4F0-E789B14BB857` | Readings in the sociology of migration | Readings in the sociology of migration | 0.7647 | 0.7647 | 0.0000 |
