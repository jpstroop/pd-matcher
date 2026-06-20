# The learned matcher

The **learned combiner** is a gradient-boosted model (LightGBM) that scores a
`(MARC record, CCE registration)` pair from the same per-scorer evidence the
rule-based pipeline already produces. It replaces the weighted mean's hand-tuned
field averaging with a function fit to the labeled vault, and it emits a
**calibrated match probability directly** — no separate Platt step.

It is built, wired, and validated. It is **not** the shipped default: the
zero-dependency [weighted mean](DESIGN.md) is, because it needs no ML libraries
and is what bootstraps labeling. Reach for the learned combiner when you want the
most accurate scores or any threshold-based triage.

For the surrounding algorithm see [DESIGN.md](DESIGN.md); for retrieval-vs-scoring
see [MATCHING_ARCHITECTURE.md](MATCHING_ARCHITECTURE.md).

---

## Why use it over the weighted mean

- **Better separation.** On a (deliberately hard, middle-heavy) held-out sample
  the learned combiner reaches ROC-AUC ≈ 0.95 versus the weighted mean's ≈ 0.94;
  in-vault GroupKFold out-of-fold AUC is ≈ 0.997. The harness is
  `scripts/separation_wild_test.py` (issue #84). Top-1 linkage precision/recall
  is saturated for both combiners and is *not* the discriminating metric — what
  matters for scaling is **separation** (can a score threshold auto-decide a
  pair), and that is where the learned model wins.
- **Calibrated output.** The score is `predict_proba`, a probability where 0.5
  genuinely means 50/50. The weighted mean's `calibrated` is `raw / 100` (a
  linear pass-through) unless a Platt calibrator artifact is present, so its
  mid-range scores are not probabilities — you cannot threshold it the way you
  can the learned model. This is why the learned combiner is the instrument for
  any auto-accept / auto-reject triage.
- **Same evidence, no second pipeline.** It reads the identical `Evidence`
  objects the scorers emit, so improving any scorer's discrimination improves
  *both* combiners at once.

## How it works

Each non-`unsure` vault pair is scored once into per-scorer `Evidence`, then
projected into a fixed-shape feature row (currently **51 features**) by
[`pd_matcher.match.combiners.features.feature_row`](../src/pd_matcher/match/combiners/features.py).
The columns are the per-scorer normalized scores plus derived sub-features, e.g.:

- `title.token_set` + `title.coverage` (containment, for asymmetric titles)
- `name.author` / `name.publisher` (IDF-gated)
- `extent.page_count`
- `volume.compat` + `volume.incompatible_uncorroborated`
  (a whole/part incompatibility not vetoed by an exact identifier)
- `lccn.exact`, `isbn.exact`, `edition.compat`
- `*__skipped` presence flags

(`year.delta` is intentionally **not** a scoring feature — exact-year retrieval
bucketing makes it a constant; see issue #88.)

The model is a `LGBMClassifier` with a **locked recipe**: `max_depth=3`,
`num_leaves=8`, `min_data_in_leaf=10`, `reg_lambda=1.0`, `n_estimators=200`,
`class_weight="balanced"`, `objective="binary"`, fixed `random_state`. Locked so
that re-training is deterministic and comparable across vault snapshots.

Honest evaluation uses **GroupKFold by `marc_control_id`** — every pair is scored
by a model that never trained on its MARC — so the reported AUC is not memorized.

## Install the ML extra

LightGBM and scikit-learn are in the optional `ml` dependency group, not the base
install:

```bash
pdm install --group ml
```

On macOS Apple Silicon, LightGBM needs `libomp` at runtime. Easiest:
`brew install libomp`. No-brew workaround — point the loader at the
scikit-learn-bundled copy:

```bash
DYLD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.14/site-packages/sklearn/.dylibs:$DYLD_LIBRARY_PATH" \
    pdm run pd-matcher train-scorer --index caches/cce.lmdb
```

## Train it

```bash
pdm run pd-matcher train-scorer --index caches/cce.lmdb
```

Defaults: `--vault data/label_vault.jsonl`, `--pool data/candidates`,
`--out-dir` is the index's parent (`caches/`). It scores every non-`unsure` vault
pair through the production pipeline, fits the locked model, prints the 5-fold OOF
AUC, and writes two artifacts:

- `caches/learned_scorer.txt` — the LightGBM model
- `caches/learned_scorer.msgpack` — feature metadata

These live under `caches/` and are **gitignored** (regenerable, never committed).
**Retrain whenever the vault grows meaningfully or a scorer's feature set
changes** — the feature row is part of the model contract, so a scorer change
makes the on-disk artifact stale.

## Use it

Select the learned combiner with `--scorer learned` (or set `scorer: learned` in
the matching config). It loads the artifact from the index's parent and **fails
loudly**, naming `train-scorer`, if no artifact is present.

```bash
# matching
pdm run pd-matcher match --scorer learned ...

# evaluation against the vault
pdm run pd-matcher eval --scorer learned ...

# held-out separation comparison (both arms)
pdm run python scripts/separation_wild_test.py
```

## Caveats

- **Not the default.** Switching the shipped default to `learned` is a real
  option now that it is validated and wired, but it is a deliberate decision (it
  adds the `ml` dependency to the production path) and has not been made.
- **The held-out sample is deliberately hard.** It is a middle-heavy,
  disagreement-weighted slice, so the ≈0.95 AUC is a conservative floor, not the
  rate on representative acquisition (which skews toward the easy tails).
- **Triage is viable, not yet deployed.** The calibrated output is what makes
  auto-accept/auto-reject thresholds *possible*; the published artifact is still
  fully human-verified (the matcher's job is candidate surfacing).
- **Vault size is the bottleneck.** At ~2,000 labels the model is solid but
  separation gains now come mostly from per-scorer feature quality and from
  growing the labeled corpus.

## History

This combiner began as a read-only diagnostic (see the now-superseded
[LEARNED_SCORER_DIAGNOSTIC.md](LEARNED_SCORER_DIAGNOSTIC.md)) used to check
whether the hand-tuned weights matched what the data implied. Once the vault
crossed ~1,500 labels and the held-out separation test (#84) confirmed it
generalizes off-vault, it graduated to a production-capable combiner.
