# Learned-scorer production A/B — 2026-06-12

The authoritative test for issue #4: the full eval pipeline run twice over the labeled vault (1,500 labels; 1,434 trainable pairs after excluding 66 unsure), once with `--scorer weighted_mean`, once with `--scorer learned`. The learned model was trained by `pdm run pd-matcher train-scorer` on the full trainable vault (53 features, locked hyperparameters from the tightening round, 5-fold OOF AUC 0.9940 at train time). Branch: `phase-4-learned-scorer`; weighted-mean reference baseline refreshed at `d6afb52`.

## Restated adoption gate

Ticket #4's original "+2 F1 on the regression eval" bar predates #71 (top-1 F1 was ~0.91; it is now ~0.99), making +2 points arithmetically impossible. Restated:

1. Top-1 linkage P/R/F1 ≥ weighted mean (no regression beyond noise).
2. Pair-level discrimination clearly superior, on honest (out-of-fold) evidence.
3. Match throughput ≥ ~70% of weighted-mean throughput.

## Results

### Pass A — top-1 linkage (the metric that matters)

| | weighted mean | learned | delta |
|:---|---:|---:|---:|
| MARCs evaluated | 922 | 922 | |
| with top pick above floor | 920 | 921 | +1 |
| correct top pick | 915 | 911 | **−4** |
| precision | 0.99457 | 0.98914 | −0.0054 |
| recall | 0.99241 | 0.98807 | −0.0043 |
| F1 | 0.99349 | 0.98861 | **−0.0049** |

### Pass B — pair-level discrimination (INVALID as a comparison on this run)

| | weighted mean | learned |
|:---|---:|---:|
| AUC | 0.9419 | 0.99995 |
| AP | 0.9677 | 0.99997 |

The learned model's near-perfect pass-B numbers are a **symptom of train-set evaluation, not generalization**: `train-scorer` fits on all trainable pairs and pass B scores those same pairs. The honest pair-level evidence is the out-of-fold record: train-time OOF AUC 0.9940, and the tightening round's OOF best-F1 0.9720 vs the weighted mean's 0.8915 on identical pairs. Pair-level superiority is real — but it is established by the OOF numbers, not by this pass.

### Throughput

Wall clock for the full eval: weighted mean 944.7 s, learned 1109.4 s → learned runs at **85%** of weighted-mean speed (bar: ≥70%). Per-candidate `Booster.predict` overhead is acceptable; no batching work needed.

## Why the model loses at top-1 while winning at pair level

Top-1 linkage is a *ranking* task over every retrieved candidate for a MARC — typically thousands of same-year, token-sharing decoys, almost all unlabeled. The training set contains only labeled pairs: the matcher's reviewed top picks (positives, plus human-rejected hard negatives). The rank-2..N decoy population is a distribution the model has never seen. At inference it over-scores some decoys that resemble labeled positives in feature space, letting them out-rank the true match. The weighted mean — a fixed formula with no training distribution — treats decoys and labeled pairs identically and is immune to this shift.

The decisive detail: the learned model loses top-1 (911 vs 915 correct) **despite having effectively memorized the labeled pairs**. Its pass-A numbers here are an upper bound; an honestly held-out top-1 eval would be the same or worse. No additional compute is needed to reach the verdict.

## Gate verdict

1. Top-1 linkage ≥ weighted mean: **FAIL** (−0.49 F1 points, with the memorization advantage).
2. Pair-level superiority (honest OOF): **PASS** (0.9940 OOF AUC vs 0.9419; +0.08 OOF best-F1).
3. Throughput ≥ 70%: **PASS** (85%).

**Disposition: do not flip the default.** `scorer: weighted_mean` remains production. Per ticket #4's own contingency: "otherwise keep it available but disabled and document why" — this document is the why.

## The path forward (follow-up ticket)

Ticket #4's original design specified training negatives as "same-year-bucket non-matches" — sampled decoys — which the research rounds replaced with labeled negatives only. This A/B shows that choice is exactly what fails at ranking. Follow-up experiment: augment training with sampled decoy negatives (for each labeled-match MARC, sample k retrieved candidates ≠ the true match as negatives; tolerate the small label-noise risk that an unlabeled candidate is a true duplicate registration), retrain, re-run this A/B. The apparatus built this phase (train-scorer, eval --scorer, the artifact format) makes that a config-and-data experiment, not an engineering one.

Secondary value shipped regardless of the verdict: the persisted model artifact is the prerequisite for the #76 audit queue (model-vs-vault disagreement review), which already proved its worth by catching 3 label errors during the research rounds.

## Reproduction

```
pdm run pd-matcher train-scorer --index caches/cce.lmdb
pdm run pd-matcher eval --index caches/cce.lmdb --scorer weighted_mean --report /tmp/ab_weighted.json
pdm run pd-matcher eval --index caches/cce.lmdb --scorer learned --report /tmp/ab_learned.json
```
