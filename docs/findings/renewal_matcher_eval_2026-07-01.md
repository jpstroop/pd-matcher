# First renewal matcher: trained vs weighted-mean, honest held-out eval

**Date:** 2026-07-01
**Gates:** issue [#45](https://github.com/jpstroop/pd-matcher/issues/45) — does the harvested
MARC↔renewal data produce a working renewal matcher, and does training beat the untrained
weighted-mean baseline?
**Scope:** STANDALONE, directional. No production code changed; the production combiner
(`match/combiners/features.py`) is untouched. Read-only; nothing under `data/` is written.
**Proof:** `scripts/renewal_matcher_eval.py` (committed). Features come from the production renewal
scorers (`title` / `name.author` / `name.publisher` / `year`) exactly as `score_renewal` in
`build_renewal_queue` wires them; the baseline is the same weighted-mean combiner
`build-renewal-queue` uses.

## TL;DR / VERDICT

Training a logistic regression on the harvested pairs **looks like a big win on the harvested set
(grouped-CV AUC 0.95 vs 0.88) but LOSES to the untrained weighted-mean baseline on the honest,
human-labeled vault test (AUC 0.62 vs 0.77).** The harvested set as currently constructed does
**not** produce a renewal matcher that beats the baseline where it counts. The trained model
overfits the harvested distribution (easy transitivity positives + same-MARC look-alike negatives)
and transfers *worse* than doing nothing.

**Recommendation: do not integrate a trained renewal matcher. Keep the weighted-mean combiner for
the renewal pathway.** The next lever is *more honest training signal* (human-labeled renewal
verdicts), not a better model on this data. Detail in "Next step" below.

## Setup

Two datasets, one shared feature extractor.

**Feature vector (8 dims per pair).** For each of the four renewal scorers — title, author,
claimants-vs-publisher, `odat`-year-vs-MARC-year — two features: the normalized `[0,1]` reading and
a `present` flag (0 when the scorer skipped, i.e. an input was absent). Skip-aware so the model can
tell "present but disagrees" (norm 0, present 1) from "no signal" (norm 0, present 0). These are the
per-scorer readings the weighted-mean combiner averages — *not* the pre-combined score.

**Baseline.** The weighted-mean renewal combiner's calibrated score on the same four Evidence
objects — the exact scorer `build-renewal-queue` ships. No calibrator is present in `caches/`, so
calibrated = raw/100 (identical to how the harvest itself scored). Untrained; only the fold/test
membership varies it.

**Trained model.** A standardized, L2-regularized **logistic regression** (`StandardScaler` +
`LogisticRegression(C=1.0)`). Chosen deliberately over a tree ensemble: with ~440 harvested rows and
8 features, a linear L2 model has far less capacity to memorize the harvested quirks, and its
coefficients are directly inspectable. (A LightGBM run was not needed — the linear model already
overfits the *distribution*, not the *sample*; more capacity would only widen the harvested-to-vault
gap.)

**Harvested set** (`data/harvested_renewal_pairs.jsonl`): 440 rows = 220 verified-by-transitivity
positives + 220 same-MARC hard-negative look-alikes, over 220 distinct MARC records. MARC and
renewal fields reconstructed straight from the JSONL (self-contained, no pool lookup).

**Vault external test** (`data/training/label_vault.jsonl`, `match_source == "renewal"`): the
human-labeled renewal verdicts, resolved MARC-from-pool + renewal-from-index. 134 renewal entries →
3 `unsure` dropped → 131 considered → **131 resolved (0 missing in pool, 0 missing in index)**:
**117 match / 14 no_match**. Never used in training. This is the honest signal.

## No-leakage protocol

- **Harvested CV is grouped by MARC control id** (`GroupKFold`, 5 splits). A MARC and its own hard
  negatives always fall on the same side of the split, so the model can never "recognize" a test
  MARC from its training-side twin.
- **The vault set is scored only after the model is frozen** on all 440 harvested rows. It shares no
  rows — and, being a different sampling process, a different distribution — with training.
- **The F1 threshold is chosen on the harvested set** and applied unchanged to the vault (a
  legitimate held-out thresholding protocol).

## Results

### Harvested set — grouped 5-fold CV (group = MARC)

| Arm | AUC per fold | mean ± sd |
|---|---|---|
| **Trained** | 0.893, 0.972, 0.946, 0.955, 0.966 | **0.9465 ± 0.0281** |
| **Baseline (weighted-mean)** | 0.805, 0.876, 0.902, 0.894, 0.919 | **0.8790 ± 0.0395** |

Resubstitution (full-train) AUC = **0.9500**, essentially equal to the 0.9465 CV mean → the model is
**not** overfitting the harvested *sample*. On the harvested distribution, training adds ~+0.07 AUC.

Baseline-fidelity check: the recomputed weighted-mean baseline correlates r=0.917 with the `score`
the harvest originally wrote. It is <1.0 because the harvest scored the *full* pool MARC while this
eval reconstructs only the single `marc_title` / `marc_author` fields the JSONL stores (no separate
`title_main` / `statement_of_responsibility`). Good enough for a fair comparison; both arms see the
identical reconstructed inputs.

### Vault external test — human-labeled, never trained on (117 match / 14 no_match)

| Arm | AUC | P/R at held-out threshold |
|---|---|---|
| **Trained** | **0.6215** | thr 0.365 → P 0.893 / R 1.000 (tp 117, fp 14, fn 0, tn 0) |
| **Baseline (weighted-mean)** | **0.7747** | thr 0.318 → P 0.892 / R 0.991 (tp 116, fp 14, fn 1, tn 0) |

**The ranking flips.** The arm that won by +0.07 on the harvested set *loses by −0.15* on the honest
test. The baseline separates the human-labeled matches from the look-alikes better than the trained
model does.

At the F1-optimal threshold (chosen on the balanced 50/50 harvested set) neither arm rejects any of
the 14 vault negatives (tn 0) — the threshold lands low, and on the 117/14-imbalanced vault both
classify nearly everything as a match. So the *thresholded* P/R is uninformative here; **AUC (rank
separation, threshold-free) is the trustworthy comparison**, and it favors the baseline.

### Why the trained model transfers worse — coefficients

Standardized LR weights: `title_norm +2.52`, `author_norm +1.13`, `author_present −1.02`,
`title_present −0.56`, `claimants_norm −0.15`, `year_norm +0.05`, `year_present 0.00`, intercept
+0.45. The model learned to lean almost entirely on **title and author agreement** — because that is
exactly what discriminates a same-MARC hard negative (a *different* renewal for the same book, whose
title/author differ) from the true positive (near-identical title/author). That rule is razor-sharp
on the harvested construction and dull on the real vault distribution, where the negatives are not
same-MARC twins and the claimants/year signals carry more of the load. The baseline's fixed weights,
which never over-committed to title/author, generalize better.

## Honest caveats (read these before acting)

- **Harvested positives are easy.** They come from transitivity (verified MARC↔registration +
  deterministic registration↔renewal), so the renewal's title/author are near-identical to the
  MARC's. The 0.95 harvested AUC overstates real-world difficulty for *both* arms.
- **Harvested negatives measure discrimination, not recall.** They are the top-scoring same-MARC
  look-alike renewals — they test "can you reject a confusable sibling," not "can you find the true
  renewal in the corpus." Corpus recall is a different, unmeasured question.
- **Small, imbalanced vault test.** 131 rows with only **14 negatives**. AUC estimated against 14
  negatives has a wide confidence interval; the −0.15 gap is directionally clear but should not be
  quoted to three digits as a stable effect size. More labeled renewal *no_match* verdicts would
  tighten this considerably.
- **Distribution shift is the whole story.** The harvested and vault sets are sampled by different
  processes; the eval's value is precisely that it exposes the shift instead of hiding it behind a
  single in-distribution split.
- **Field reconstruction shortcut** on the harvested side (r=0.917 fidelity) — a minor,
  symmetric approximation applied to both arms.

## Does training beat baseline?

**No — not on the signal that matters.** It wins on the harvested set (an in-distribution,
easy-positive / twin-negative construction) and loses on the human-labeled vault (AUC 0.62 vs 0.77).
A legitimate and useful finding: the untrained weighted-mean renewal combiner is already the
stronger renewal matcher on real labels, so there is no reason to add model complexity now.

## Next step

1. **Ship nothing to production.** Keep the weighted-mean combiner on the renewal pathway; do not
   wire a trained renewal model into `match/combiners/features.py`. No artifact was saved (a model
   that loses the honest test is not worth keeping).
2. **Grow honest renewal training signal.** The bottleneck is data construction, not model choice.
   Transitivity-harvested pairs are too easy and their hard-negatives too artificial to teach a
   generalizing matcher. Gather more *human-labeled* renewal verdicts (especially `no_match`s — the
   vault has only 14) via the renewal review queue, then re-run this exact eval. The harvested set is
   still useful as **augmentation / pretraining** under heavy regularization, but only if a trained
   arm can be shown to match or beat the weighted-mean baseline *on the vault*, which today it cannot.
3. **Re-open the model question only after** the vault renewal set is materially larger and more
   balanced; at that point re-measure with this same grouped-CV + external-vault protocol before any
   integration.
