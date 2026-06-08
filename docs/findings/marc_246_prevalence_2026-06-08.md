# MARC 246 prevalence audit (issue #73)

Generated: 2026-06-08T20:50:57.368396+00:00

## Method

Walks every ``<record>`` element under `/Users/jstroop/workspace/public_domain/data/candidates` (69 `<lang>/*.xml` and `<lang>/*.marcxml` shards) via `lxml.etree.iterparse`, tallying every `<datafield tag="246">` child by its `ind2` attribute. Blank / missing `ind2` is bucketed as `_`. The matcher's `iter_marc_records()` parser is deliberately bypassed because it does not extract 246. The CCE-likely `ind2` set is {`0`, `2`, `3`, `4`, `7`, `8`}: portion-of-title, distinctive, other, cover, running, and spine titles — the variant-form flavors most likely to match the title strings CCE catalogers transcribed. Parallel (`1`), added-title-page (`5`), caption (`6`), and blank are excluded as too noisy or too distant from the CCE-side title surface.

## Headline

- Total records scanned: 334,954
- Records with at least one 246: 5,651 (1.687%)
- Total 246 datafields: 6,852
- Mean 246-per-record among records with any 246: 1.213

## Distribution by `ind2`

| ind2 | label | records with >=1 | pct of corpus | total occurrences | CCE-likely |
| --- | --- | ---: | ---: | ---: | :---: |
| `_` | _ — no information / blank | 2,954 | 0.882 | 3,383 | no |
| `0` | 0 — portion of title | 1,268 | 0.379 | 1,472 | yes |
| `4` | 4 — cover title | 869 | 0.259 | 884 | yes |
| `1` | 1 — parallel title | 295 | 0.088 | 369 | no |
| `3` | 3 — other title | 233 | 0.070 | 285 | yes |
| `8` | 8 — spine title | 214 | 0.064 | 214 | yes |
| `5` | 5 — added title page title | 122 | 0.036 | 131 | no |
| `6` | 6 — caption title | 71 | 0.021 | 71 | no |
| `7` | 7 — running title | 28 | 0.008 | 28 | yes |
| `2` | 2 — distinctive title | 15 | 0.004 | 15 | yes |

## CCE-likely subtotal

Records with at least one 246 whose `ind2` is in {`0`, `2`, `3`, `4`, `7`, `8`}: **2,493** (**0.744%** of 334,954 total records).

## Distribution of 246-count per record (among carriers)

| 246 datafields | records |
| --- | ---: |
| 1 | 4,739 |
| 2 | 715 |
| 3 | 158 |
| 4 | 24 |
| 5+ | 15 |

## Decision

- CCE-likely records: 2,493 (0.744%) of 334,954 corpus records
- Threshold: >2.0%

Numbers do not justify follow-up; recommend closing #73 with this finding as durable record.
