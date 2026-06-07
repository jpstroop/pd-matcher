# Floor sweep — 2026-06-07

## 1. Experimental setup

Sweep `min_combined_score` against the labeled vault MATCH set. Runs the production candidate retrieval + scoring pipeline once per MARC with `calibrator=None` (linear pass-through per #70), records `(gt_rank, gt_calibrated, winner_calibrated)`, then varies the floor in post-processing to produce a precision/recall/F1 curve at no extra cost.

- **Run date** (UTC): 2026-06-07
- **Vault**: `data/label_vault.jsonl`
- **Candidate pool**: `data/candidates`
- **Index**: `caches/cce.lmdb`
- **Labeled MATCH MARCs resolved**: 836
- **`year_window`**: 0
- **Calibrator**: none (linear pass-through, `calibrated = raw / 100`)

## 2. Sweep table

Recall denominator is the count of resolved MATCH MARCs (`agree_above + agree_below + out_ranked_above + out_ranked_below + missing`). Precision denominator is `marcs_with_top` (the count of MARCs where the *winner's* calibrated score clears the floor, regardless of whether the winner is the GT or a wrong record). `agree_above` is the count of MARCs where the GT is rank 1 AND its calibrated score clears the floor — i.e. recall.

| floor | marcs_with_top | agree_above | agree_below | out_ranked_above | out_ranked_below | precision | recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 834 | 833 | 2 | 1 | 0 | 0.9988 | 0.9964 | 0.9976 |
| 0.55 | 832 | 831 | 4 | 1 | 0 | 0.9988 | 0.9940 | 0.9964 |
| 0.60 | 832 | 831 | 4 | 1 | 0 | 0.9988 | 0.9940 | 0.9964 |
| 0.65 | 781 | 780 | 55 | 1 | 0 | 0.9987 | 0.9330 | 0.9647 |
| 0.70 | 712 | 711 | 124 | 1 | 0 | 0.9986 | 0.8505 | 0.9186 |

## 3. Deltas from the production floor (0.70)

Δ values are `(floor) - (0.70)`. Positive Δ on recall means the floor reduction recovers MATCH MARCs that were previously floor-suppressed. Negative Δ on precision is the cost — wrong-winner MARCs whose winner score is in the relaxed band now count as predictions.

| floor | Δ precision | Δ recall | Δ F1 |
|---:|---:|---:|---:|
| 0.50 | +0.0002 | +0.1459 | +0.0790 |
| 0.55 | +0.0002 | +0.1435 | +0.0778 |
| 0.60 | +0.0002 | +0.1435 | +0.0778 |
| 0.65 | +0.0001 | +0.0825 | +0.0461 |
| 0.70 | +0.0000 | +0.0000 | +0.0000 |

## 4. Recommendation

By F1, the highest-scoring floor is **0.50** (P=0.9988 R=0.9964 F1=0.9976). The production setting is `0.70`.

Reading the sweep table is more informative than a single point pick: for a recall-priority application (the matcher publishes a linkage and downstream consumers apply their own PD reasoning per the project's stated purpose), recall improvements are likely worth modest precision drops. For a precision-priority application the production floor stays.
