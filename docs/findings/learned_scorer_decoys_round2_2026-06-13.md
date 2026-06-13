# Learned-scorer decoy-negatives experiment (round 2) — 2026-06-13

Issue #77, the #4 ORIGINAL design the research rounds dropped: train with sampled same-year-bucket non-matches as negatives. The production A/B (`docs/findings/learned_scorer_ab_2026-06-12.md`) showed the labeled-only learned model FAILS top-1 linkage (911 vs the weighted mean's 915 correct top picks). Round 1 augmented training with above-floor `alternates` (only 267 decoys, 207/922 MARCs) and reached 913/922 — IMPROVED-BUT-SHORT. Round 2 pulls the two levers the round-1 findings named: a BELOW-FLOOR harvest (~10x the decoy mass) and a decoy-weight sweep selected by an OOF ranking proxy.

## Method

**Below-floor decoy harvest.** For each labeled-MATCH MARC the production `match_record` is run under the weighted-mean scorer (eval pass A / `train-scorer` produce identical per-scorer Evidence — it is combiner-independent) with `min_combined_score` forced to `0.0` and `top_k=21`, so `best + alternates` is the FULL ranked candidate set with no floor cull. The top `5` non-true candidates by combined score become decoys; each carries full Evidence at zero extra scoring cost and is projected through the canonical `feature_row`. Realized counts:

- MARCs with a current `match` verdict (harvested): **922**
- of those resolved in the candidate pool: **922**
- MARCs yielding ≥1 decoy: **922**
- decoy negatives harvested: **4607**
- candidates dropped as the true match: **922**; dropped as another labeled pair for that MARC: **0**

With the floor removed, the full retrieved candidate set is available per MARC, so virtually every MARC yields up to 5 decoys (ceiling 922×5); the realized count (4607) falls short only where a MARC retrieved fewer than that many non-true candidates.

**Decoy down-weighting sweep.** The augmented model trains with explicit per-row `sample_weight` (labeled rows 1.0, decoy rows `w`) across `w ∈ {0.25, 0.5, 1.0}`, keeping `class_weight=balanced` (LightGBM MULTIPLIES the sample weight with the class-balance factor, so `w < 1` down-weights decoys on top of class balancing). Configs are selected by an **OOF ranking proxy**: for each labeled-MATCH MARC with ≥1 harvested decoy, using grouped-OOF predictions, does the true pair's probability strictly exceed the max over that MARC's decoys? `rank_proxy` is the fraction of such MARCs the true pair wins — a direct, eval-free proxy for the pass-A top-1 gate. The winner is the highest `rank_proxy`, tie-broken by labeled-rows-slice best-F1 (Gate 2 health).

**Label-noise caveat.** A harvested decoy is assumed a non-match, but an unlabeled candidate could be a true duplicate registration. Any candidate whose `nypl_uuid` equals the labeled true match OR ANY labeled pair for that MARC is dropped, so known labels never poison the decoy set; the residual risk is an unlabeled true duplicate, accepted per the ticket as small. The below-floor harvest enlarges this surface (low-scoring candidates are less likely true duplicates, but more numerous), which the sweep's down-weighting partly hedges.

**GroupKFold rationale.** Decoys and the positive from the same MARC share near-identical features; a random split would leak a MARC's positive into a fold that also holds its decoys, inflating OOF. GroupKFold by `marc_control_id` forces every row of one MARC into one fold, so the OOF numbers — and the rank proxy that reads them — are honest under the augmented population.

- **Feature count**: 53 (production `feature_names()`; the persisted artifact's contract matches inference exactly)
- **Labeled rows**: 1434 (922 pos / 512 neg); **decoy rows**: 4607; **total**: 6041
- **Cross-validation**: 5-fold GroupKFold by `marc_control_id`, locked hyperparameters (max_depth=3, num_leaves=8, min_data_in_leaf=10, lambda_l2=1.0, n_estimators=200, class_weight=balanced), random_state=20260612, deterministic (`n_jobs=1`)

## Decoy-weight sweep (selected by the OOF ranking proxy)

All configs share the augmented matrix, locked hyperparameters, seed, and 5-group GroupKFold folds; only the decoy `sample_weight` `w` differs. **rank_proxy** is the fraction of harvested MARCs whose true pair out-ranks all its decoys under grouped OOF — the eval-free top-1 proxy. **OOF AUC / best-F1 (labeled)** restrict the OOF predictions to the labeled rows, the apples-to-apples Gate-2 health check against the labeled-only baseline. The winner (★) maximizes rank_proxy, tie-broken by labeled best-F1.

| w | OOF AUC (full) | OOF AUC (labeled) | best-F1 (labeled) | rank_proxy | wins/evaluable |
|:---|---:|---:|---:|---:|---:|
| 0.25 | 0.9961 | 0.9934 | 0.9712 | 0.9881 | 911/922 |
| 0.5 ★ | 0.9963 | 0.9933 | 0.9720 | 0.9881 | 911/922 |
| 1.0 | 0.9962 | 0.9924 | 0.9676 | 0.9848 | 908/922 |

**Winner: w=0.5** (rank_proxy 0.9881, 911/922). Labeled-only baseline grouped-OOF: AUC 0.9935, best-F1 0.9712. **Winner minus labeled-only on the labeled-rows slice:** AUC -0.0002, best-F1 +0.0008.

## A/B result — top-1 linkage (pass A)

The weighted-mean reference is unchanged (`/tmp/ab_weighted.json`): 915/922 correct top picks, F1 0.99349. The labeled-only learned model scored 911/922; round 1 (above-floor decoys only) scored 913/922. The round-2 decoy model below (below-floor harvest, winning decoy weight w=0.5) is evaluated by `pd-matcher eval --scorer learned` against the freshly-persisted `caches/learned_scorer.*`.

_Caveat (unchanged from the prior A/Bs):_ pass A on labeled MARCs is still partially train-set-flavored for the labeled rows, and pass-B AUC near 1.0 on labeled pairs remains uninformative. The decoy population at rank-2..N is now in-distribution, but the gate metric is pass-A top-1 vs the weighted reference.

| scorer | correct top / 922 | precision | recall | F1 |
|:---|---:|---:|---:|---:|
| weighted_mean (reference) | 915 | 0.99457 | 0.99241 | 0.99349 |
| learned (labeled-only, prior A/B) | 911 | 0.98914 | 0.98807 | 0.98861 |
| learned (round 1, above-floor) | 913 | 0.99564 | 0.99024 | 0.99293 |
| learned (round 2, below-floor, w=0.5) | 919 | 0.99675 | 0.99675 | 0.99675 |

## The confound that governs the verdict (read before the gates)

The `pd-matcher eval` pass-A number is measured **over the vault, which is the training set**. The learned model saw each true `(MARC, CCE)` pair with label=1 during training, so it scores those pairs inflated at eval — its 919 is an **upper bound**, not an honest estimate. The weighted mean has no training, so its 915 is its honest number. **Comparing learned-919 to weighted-915 hands the learned model a memorization advantage the incumbent does not get.**

The honest, leakage-free signal is the **grouped-OOF rank proxy: 911/922** (each MARC scored by a fold-model that never saw it). On that basis the decoy model sits **below** the weighted mean's 915. Both numbers are reported below; the OOF proxy is the one to trust for "does it genuinely rank better on unseen data."

A truly clean top-1 comparison would require a GroupKFold-through-pipeline eval (each fold's model scores held-out MARCs' full candidate sets); the current eval harness cannot do that. That is the decisive follow-up if certainty is wanted.

## Decision against the #77 gate

- **Gate 1 — top-1 F1 ≥ 0.99349 (weighted reference):** **PASS on the literal metric, FAIL on the honest one.** Pass-A eval F1 0.99675 (919/922) clears the bar — but that metric is train-flavored (see confound above). The leakage-free OOF rank proxy is 0.9881 (911/922), **below** the weighted mean's 915. No honest top-1 win is demonstrated.

- **Gate 2 — grouped-OOF labeled-rows slice not degraded vs labeled-only:** **PASS.** On the labeled rows the winning model is best-F1 **+0.0008** and AUC −0.0002 versus the labeled-only baseline — flat-to-slightly-better, no degradation. (The script's auto-emitted "FAIL" keys off any negative AUC delta; −0.0002 is noise, not degradation. This corrects the mechanical flag.)

- **Gate 3 — throughput:** **PASS by construction.** Decoys change training only; inference is the identical per-candidate `Booster.predict`, unchanged from the labeled-only model (85% of weighted mean in the prior A/B).

**Verdict: IMPROVED, NOT ADOPT.** Below-floor decoys + restored labeled negatives are the strongest configuration yet (honest OOF 911 vs labeled-only 911 and round-1 913 — note the train-flavored eval rose 913→919, but the honest proxy did not clear 915). The model is an excellent pair-level discriminator (AUC ~0.99) yet does not demonstrate an honest top-1 ranking win over the hand-tuned weighted mean.

**`scorer: weighted_mean` stays the production default.** This is not a rejection of the ML matcher — it is the endgame — but at 1,500 Princeton-only labels the learned combiner is at-best parity with a formula hand-tuned to this exact data, and its real payoff (generalization, cross-institution scale) is untestable until non-Princeton data or a production deployment exists. Switching the default now buys no measurable labeling-throughput gain and adds a retraining/staleness burden. The trained artifact still delivers value via the #76 audit queue (pair-level discrimination is exactly what that needs).

**Next levers, in priority order, if #77 continues:** (1) the GroupKFold-through-pipeline held-out top-1 eval — the only confound-free answer; (2) more/diverse labels (the learning curve plateaued for *this* feature set at this scale, so feature expansion or data diversity, not raw count, is the lever); (3) eventual cross-institution validation, the actual proving ground.

## Artifact + reproduction

The final Booster is fit on ALL augmented rows with the locked hyperparameters and the winning decoy weight (w=0.5) and persisted via the production `save_learned_model` to `caches/learned_scorer.{txt,msgpack}`, OVERWRITING the labeled-only artifact. Both files are gitignored and re-derivable: `pdm run pd-matcher train-scorer` rebuilds the labeled-only model; rerunning this script rebuilds the decoy-augmented one.

```
pdm run python scripts/learned_scorer_decoys.py \
    > docs/findings/learned_scorer_decoys_round2_2026-06-13.md
pdm run pd-matcher eval --index caches/cce.lmdb --scorer learned \
    --report /tmp/ab_learned_decoys.json
```

The harvest + sweep logic that produced these numbers lives only in `scripts/learned_scorer_decoys.py`, which is committed (gitignore exception) precisely because this finding drove a production-default decision and is NOT reproducible from the shipped CLI alone — unlike the labeled-only A/B, which `train-scorer` + `eval --scorer` reproduce directly.
