# Field-pairing study

## Context

`pd-matcher` compares a MARC record against Catalog of Copyright Entries (CCE)
registrations through per-field fuzzy scorers (title, author, publisher). The
catch is *transposition*: the two sources do not always put the same value in
the same field. A publisher records the work title where the series title
belongs; the copyright claimant carries what a cataloguer would call the
publisher; the author appears only in the 245 statement of responsibility
because there is no 1xx author field. A single fixed pairing (`marc.title ↔
cce.title`, `marc.main_author ↔ cce.author_name`, …) misses these.

The matcher already tried *some* alternate pairings, but they were hard-coded
illustrative guesses (title↔series, publisher↔claimants) and the author scorer
had no alternate pairing at all. This study lands a **configurable** field-pairing
subsystem — the pairing set is now `config/defaults/field_pairings.yaml`, tuned
by editing config and re-running `eval`, not by editing code — and confirms the
data-derived default set does not regress precision while recovering recall.

See [design.md §Step 3](../design.md) for the architecture (code surfaces raw
subfields; a closed combine vocabulary in YAML composes and pairs them; every
name is validated at compile time).

## The pairing set under test

| Group | MARC field | CCE field | Recovers |
|---|---|---|---|
| title | `title` (fused $a+$b) | `title` | the normal case |
| title | `title_main` ($a only) | `title` | $b subtitle noise the CCE title lacks |
| title | first `series_titles` | `title` | work title stored as a series title |
| author | `main_author` | `author_name` | the normal case |
| author | `statement_of_responsibility` | `author_name` | no 1xx; author only in 245$c |
| author | `main_author` | `claimants` | author recorded as the claimant |
| publisher | `publisher` | `publisher_names` | the normal case |
| publisher | `publisher` | `author_name` | self-published / author-as-publisher |

The pipeline scores every pairing in a group through the group's scorer and keeps
the single best Evidence per group; the losers are retained on
`CandidateMatch.losing_evidence` for audit. The combiner is unchanged — it still
sees exactly one title/author/publisher Evidence.

## Method

- Tool: `pd-matcher eval`, which runs the full match pipeline against the
  ground-truth pairings and reports precision, recall, and F1 (best predicted
  match's CCE id vs. the known-correct id).
- Same sample, seed, window, and worker count as the [year-window
  study](year-window.md) so the numbers are directly comparable: a single 500-row
  random sample (`--seed 42`), exact-year blocking (`--year-window 0`).
- The "before" numbers are the year-window study's exact-year baseline (the
  hard-coded-pairings pipeline) on the identical sample/seed.

### Caveat: thin records

As in the year-window study, the eval reconstructs each `MarcRecord` from the
bibliographic columns embedded in the ground-truth file, not from full source
MARC. Those columns omit fields the richer pairings would exercise. In
particular the GT file carries only a single, **already-fused** title string, so
its `title_main` is set equal to `title`: the `title_main ↔ title` pairing in
this eval is *not* a genuine "$a-only" test. This does not affect the comparison
(both before and after run against the same reconstructed records), but it means
the eval understates what the `$a`-isolation and 245$n/$p part-title pairings can
do on real MARC.

### Why the "drop $b is stronger" finding was an artifact

An earlier exploratory study reported that comparing the title's `$a` alone beat
the fused `$a + $b`. That result split an *already-fused* GT title string on
punctuation — it never had isolated `$a` to begin with, so it was measuring
"truncate the title at the first delimiter," not "use 245$a." The honest fix is
to stop discarding `$a` in the parser (done here: `MarcRecord` now carries
`title_main` for `$a` and `title_part_number`/`title_part_name` for `$n`/`$p`,
distinct from the fused `title`) and to test real variants on full MARC. That
honest measurement is deferred to **issue #18** (field-pairing experiments),
which this subsystem exists to make a config-edit-plus-re-eval loop.

## Results

### Exact-year, 500 rows, seed 42

| pairings | precision | recall | F1 | predicted | agreeing |
|---|---|---|---|---|---|
| before (hard-coded) | 0.863 | 0.780 | 0.819 | 452 | 390 |
| **after (configurable defaults)** | **0.872** | **0.792** | **0.830** | 454 | 396 |
| delta | +0.009 | +0.012 | +0.011 | +2 | +6 |

The configurable default set recovers six additional correct matches (390 →
396) while making only two more predictions (452 → 454), so **both** precision
and recall improve: +1.2 points recall, +0.9 points precision, +1.1 points F1.
The recovered matches come from the author group, which previously had no
alternate pairing — the `statement_of_responsibility ↔ author_name` and
`main_author ↔ claimants` pairings supply an author signal where the single
`main_author ↔ author_name` pairing was skipped or scored low.

## Decision

Ship the configurable subsystem with the data-derived default pairing set above.

- **Quality:** recall and precision both improve on the shared sample/seed;
  acceptance was "recall holds or improves and precision does not drop more than
  ~0.5pp" — precision rose.
- **Maintainability:** the pairing set is now a YAML edit. Future tuning (issue
  #18) is a config change plus a re-eval, with typos caught at load time.

## Reproduction

The original study used the pre-vault CSV-driven eval (`--ground-truth
data/combined_ground_truth.csv`, retired with #25 in 2026-05-25). The numbers
above reflect that run and won't reproduce identically against the current
vault-driven eval, which uses every labeled pair instead of a 500-pair sample.
The equivalent command today is:

```bash
pdm run pd-matcher eval \
  --vault data/label_vault.jsonl \
  --pool data/candidates \
  --index caches/cce.lmdb \
  --year-window 0 \
  --report eval_pairings.json
```

The "before" row is the exact-year (N=0) line from the
[year-window study](year-window.md), produced against the pre-change
(hard-coded-pairings) pipeline.

The pairing set in `src/pd_matcher/config/defaults/field_pairings.yaml` has
grown since this study (renewal-side pairings + publisher↔claimants added in
2026-05-27); the 8 pairings tabulated above were the set at study time.
