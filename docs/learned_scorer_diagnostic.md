# Learned-scorer diagnostic

A periodic ML-style sanity check on the hand-tuned scoring architecture in
`matching.yaml`. Trains a small LightGBM classifier on the labeled vault as a
*diagnostic instrument* — never as a production combiner replacement — to
surface mismatches between data-learned feature importance and the current
hand-tuned weights, and to produce a worktable of pairs where the learned
model and the live combiner disagree most.

For the broader matching algorithm, see [design.md](design.md). For the
labeling loop that produces the vault, see [USER_GUIDE.md](USER_GUIDE.md).

---

## Purpose

The diagnostic answers two questions:

1. **Are the hand-tuned weights in `matching.yaml` consistent with what the
   labeled corpus says matters?** LightGBM's `gain` importance, averaged
   across folds, is set alongside each scorer's current weight. Large
   divergences (a feature with high importance but a low weight, or vice
   versa) are candidates for the next round of weight inspection or for new
   scorers entirely.
2. **Which labeled pairs is the live combiner most wrong about, relative
   to what the data implies?** The top-30 disagreement table sorts pairs by
   `|lgbm_pred − combined_score|` desc. These pairs are the worktable for
   the next round of scorer/pairing additions — they typically expose
   either a missing pairing (signal the scorer architecture is throwing
   away) or a calibration miss.

The output is **directional**, not deployable. At the current corpus size
(~650 labels, severely class-imbalanced), per-fold metrics are noisy and the
model is not stable enough to ship as a combiner replacement; the
architectural pivot to a learned combiner is gated on a ~1500–2500 corpus.

## When to re-run

Re-run when the labeled corpus has grown meaningfully **or** when the scorer
architecture has changed. The original snapshot
([docs/findings/learned_scorer_diagnostic_2026-05-29.md](findings/learned_scorer_diagnostic_2026-05-29.md))
was at ~650 labels; the next planned re-run is at ~1500.

Useful re-run triggers:

- A substantial labeling milestone (each ~500–1000-label increment).
- A scorer addition or weight rework (so the diff against the prior
  diagnostic shows whether the change moved feature importance the way you
  expected).
- A new pairing (the disagreement table should shed pairs the new pairing
  fixed; pairs it didn't fix are the new worktable).

The diff between consecutive diagnostics is the most informative output —
keep prior runs in `docs/findings/` so they can be read alongside each new
one.

## How to run

LightGBM and scikit-learn are in the optional `ml` dependency group; they
are **not** in the base install. Install them once:

```bash
pdm install --group ml
```

Then run the script, redirecting the markdown report to a date-stamped
findings file:

```bash
pdm run python scripts/learned_scorer_diagnostic.py \
    > docs/findings/learned_scorer_diagnostic_YYYY-MM-DD.md
```

### libomp on macOS Apple Silicon

LightGBM links against `libomp` at runtime. The scikit-learn wheel ships its
own copy in `sklearn/.dylibs/`, but LightGBM's loader doesn't see it
automatically. Two options:

- **Easier**: `brew install libomp` once. LightGBM then finds it on the
  system load path and the script runs unchanged.
- **No-brew workaround**: prepend `DYLD_LIBRARY_PATH` so LightGBM finds the
  scikit-learn-bundled copy:

  ```bash
  DYLD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.14/site-packages/sklearn/.dylibs:$DYLD_LIBRARY_PATH" \
      pdm run python scripts/learned_scorer_diagnostic.py \
      > docs/findings/learned_scorer_diagnostic_YYYY-MM-DD.md
  ```

  Adjust `python3.14` to whatever interpreter `.venv/lib/` actually contains.

The script is read-only against the vault, the candidate pool, and the LMDB
index. It writes nothing of its own — the markdown report is stdout only.

## Outputs

The markdown report has four sections:

1. **Experimental setup**: corpus size (pair count, match/no_match split),
   the LightGBM hyperparameters, and per-fold ROC-AUC + PR-AUC mean ± std
   across the 5 stratified folds.
2. **Feature importance ranking**: each feature's normalized LightGBM `gain`
   importance averaged across folds, sorted desc, with the scorer's current
   weight from `matching.yaml` printed alongside. The `_skipped` flag
   features have no direct weight analogue and are marked `--`.
3. **Per-feature SHAP contribution distributions**: mean absolute
   contribution, standard deviation, and a coarse direction label
   (`positive` / `negative` / `bidirectional` / `inert`) across all
   out-of-fold predictions. High `std` relative to `mean_abs` signals
   interaction effects.
4. **Top-30 disagreement table**: pairs sorted by
   `|lgbm_pred − combined_score|` desc, with `pair_id`, MARC `control_id`,
   NYPL `uuid`, human verdict, combined score, LightGBM probability,
   absolute delta, and truncated MARC + CCE titles for spot-checking.

## Output convention

Append the date-stamped findings file to `docs/findings/`. Filenames follow
`learned_scorer_diagnostic_YYYY-MM-DD.md`. Keep older runs in place — the
diff between consecutive runs is the most informative thing about any
re-run, and rewriting history loses that.

## Reusable scaffold

The feature extraction itself lives in
[`src/pd_matcher/eval/feature_matrix.py`](../src/pd_matcher/eval/feature_matrix.py):
`extract_feature_matrix()` resolves every non-`unsure` vault entry to its
MARC + CCE record, runs the live scoring pipeline, and projects the winning
`Evidence` per scorer into a fixed-shape `(n, k)` feature matrix plus a
parallel `FeatureMatrixRow` tuple with provenance.

If a follow-on study wants vault-driven features (clustering, manual
inspection, an alternative learned-combiner experiment), reuse
`extract_feature_matrix()` rather than re-deriving the evidence vector.
Column order is exposed via `feature_column_names()` so callers don't have
to hardcode indices.

## Known caveats at small N

- **Fold-level CV metrics are noisy.** A few-hundred-pair corpus with a
  ~10:1 class imbalance means each fold has ~15 negatives. ROC-AUC and
  PR-AUC swing on individual labels. Treat the std-across-folds as the
  honest noise estimate, not the mean.
- **Feature importance ranking is the most stable output.** Even at small
  N, top-ranked features tend to remain top-ranked across re-runs. Use the
  *ranking* as the signal; the absolute normalized importance values are
  noisy.
- **Per-pair SHAP and probability calibration are NOT stable** at this N.
  The disagreement table is useful as a worktable for inspection, not as a
  calibration check.
- **The architecture is not deployable as a combiner replacement at this
  scale.** Replacing the weighted-mean combiner with a learned model is
  gated on roughly 1500–2500 labels with healthier negative-class
  representation. This diagnostic is the prelude, not the deployment.

The optional `ml` dependency group in
[`pyproject.toml`](../pyproject.toml) exists for exactly this script — base
installs and production paths don't pull `lightgbm` or `scikit-learn`.
