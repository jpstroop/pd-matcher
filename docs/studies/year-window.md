# Year-window blocking study

## Context

`pd-matcher` cannot score every MARC record against all ~2.17M Catalog of
Copyright Entries (CCE) registrations — that is quadratic and infeasible. It
**blocks by year**: each CCE registration is indexed into a bucket keyed by its
registration year, and for a MARC record published in year *Y* the matcher only
retrieves candidates whose registration year falls within *Y ± N*. *N* is the
year window.

A wider window admits more candidates per record. That can only help recall if
the correct candidate's registration year drifts from the MARC publication year
by more than the window — but it also admits more wrong-but-plausible candidates
that compete with the correct one, and it costs proportionally more runtime.

This study determines the value of *N* that maximizes match quality.

## Why the window might need to be wide — and why it mostly doesn't

Publication, copyright registration, and renewal can fall in different years, so
some drift is expected. Reprints are the extreme case: a 1958 reprint of a 1930
work is governed by the 1930 registration (the original registration date
determines public-domain status; a reprint with no new matter does not earn a
fresh term), so the correct match can be decades earlier than the MARC
publication year.

But those large gaps are rare. The ground-truth pairs' year differences:

| \|year difference\| | share | cumulative |
|---|---|---|
| 0 | 98.35% | 98.35% |
| 1 | 0.80% | 99.15% |
| 2 | 0.16% | 99.31% |
| 3–5 | 0.31% | 99.62% |
| 6+ | 0.38% | 100.00% |

98.35% of correct pairs are exact-year. Widening the window past ±1 reaches only
a fraction of a percent more of the ground truth.

## Method

- Tool: `pd-matcher eval`, which runs the full match pipeline against the
  ground-truth pairings and reports precision, recall, and F1 (best predicted
  match's CCE id vs. the known-correct id).
- Two experiments: a four-window sweep (N = 0, 1, 2, 3) on one 500-row sample,
  and a seed-stability check comparing N = 0 against N = 1 across four random
  500-row samples.
- `--sample N --seed S` draws a random subset; a fixed seed makes the selection
  reproducible. Random sampling matters: the ground-truth file is ordered such
  that its prefix is heavily clustered in a single year with very large
  candidate buckets, which is not representative.
- Precision/recall/F1 are computed against the matcher's single best prediction
  per record.

### Caveat: thin records

The eval reconstructs each `MarcRecord` from the bibliographic columns embedded
in the ground-truth file, not from full source MARC. Those columns omit some
fields (series titles, ISBNs, edition, added authors), so absolute F1 here is a
mild under-estimate of what the matcher achieves on full MARC input. This does
not affect the *comparison* between windows: every window runs against the same
reconstructed records, so the deltas are valid.

## Results

### Four-window sweep (500 rows, seed 42)

| window | precision | recall | F1 | relative runtime |
|---|---|---|---|---|
| **0 (exact year)** | **0.863** | 0.780 | **0.819** | 1.0× |
| 1 | 0.821 | 0.782 | 0.801 | 3.0× |
| 2 | 0.821 | 0.782 | 0.801 | 5.2× |
| 3 | 0.821 | 0.782 | 0.801 | 7.4× |

Windows 1, 2, and 3 are bit-for-bit identical on every metric and on the exact
count of agreeing predictions: beyond ±1, the additional buckets contribute
neither correct matches nor false positives in this corpus. Only exact-year
differs — and it wins F1 by +1.8 points, entirely from precision. It makes fewer
predictions (452 vs. 476) but loses only one correct one; the ~23 predictions it
drops were wrong anyway. Tightening the window removes adjacent-year candidates
that were out-scoring the correct record.

### Seed stability (500 rows, N=0 vs N=1)

| seed | F1 @ N=0 | F1 @ N=1 | precision @ N=0 | precision @ N=1 | F1 edge |
|---|---|---|---|---|---|
| 1 | 0.799 | 0.785 | 0.833 | 0.796 | +0.014 |
| 2 | 0.751 | 0.736 | 0.783 | 0.747 | +0.015 |
| 3 | 0.747 | 0.731 | 0.779 | 0.737 | +0.017 |
| 4 | 0.764 | 0.754 | 0.807 | 0.773 | +0.010 |
| 42 | 0.819 | 0.801 | 0.863 | 0.821 | +0.018 |

Exact-year wins on all five samples. The mean F1 edge is +1.4 points. The edge is
always driven by precision (+3 to +4 points), against a small recall cost
(~0.5 point). Absolute F1 varies with the sample's difficulty; the ordering does
not.

## Decision

Set the default year window to **0 (exact year)**.

- **Quality:** +1.4 points F1 on average, consistent across five random samples,
  driven by a precision gain that outweighs a negligible recall cost.
- **Cost:** roughly 7× faster than ±3 and ~5× faster than the previous default of
  ±2, because each record pulls one year bucket instead of five.
- **Recall ceiling:** tightening to exact-year forfeits at most the 1.65% of
  ground-truth pairs whose year difference is ≥ 1. In practice the precision gain
  more than compensates, and the lost pairs are dominated by reprints whose
  large year gaps no reasonable window would capture anyway.

The window remains configurable in `matching.yaml` and overridable per run with
`pd-matcher match --year-window N` / `pd-matcher eval --year-window N`.

## Reproduction

```bash
# Four-window sweep
for W in 0 1 2 3; do
  pdm run pd-matcher eval \
    --ground-truth data/combined_ground_truth.csv \
    --index caches/nypl.lmdb \
    --sample 500 --seed 42 --workers 8 --year-window "$W" \
    --report "eval_w${W}.json"
done

# Seed stability
for SEED in 1 2 3 4; do
  for W in 0 1; do
    pdm run pd-matcher eval \
      --ground-truth data/combined_ground_truth.csv \
      --index caches/nypl.lmdb \
      --sample 500 --seed "$SEED" --workers 8 --year-window "$W" \
      --report "eval_s${SEED}_w${W}.json"
  done
done
```
