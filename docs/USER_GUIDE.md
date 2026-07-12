# User guide

A guided tour of the operational picture: mental model, daily flows, maintenance. If you're looking for *what* this project is and *why* it exists, start with the [README](../README.md) — this guide picks up after you've decided to actually run it. For the matching algorithm itself, see [docs/DESIGN.md](DESIGN.md).

---

## Two things this tool does

Two distinct modes live in this repo, and most confusion comes from blurring them:

1. **Build & review a training set** — *the browser-UI workflow.* You sample candidate `(MARC, CCE)` pairs, judge each one in a local review UI, and accumulate a verified vault. That vault is the ground truth that trains and measures the matchers, and it is the bulk of day-to-day work here. Commands: `pd-matcher index build` plus the `pd-groundtruth` family (`acquire`, `build-queue`, `review`).

2. **Match a catalog to produce pairs** — *the matcher as a tool.* You run the matcher over MARC records to emit a `(MARC record, CCE registration)` linkage table — the actual "which of my books appear in these copyright records" use. Command: `pd-matcher match` (→ a JSONL file).

Both modes share one engine — the CCE index plus the per-field scorers — and differ only in what they emit: mode 1 builds a labeling queue and grows the vault; mode 2 writes a linkage table. The matcher that powers mode 2 is exactly what mode 1's vault trains and validates. The diagram below traces **mode 1**; mode 2 is a single `pd-matcher match` run.

> **Just want to match your own catalog?** If you have a MARCXML export and want the linkage table — "which of my books appear in the copyright records" — you only need **mode 2**, and you can skip the entire training/labeling apparatus (`acquire`, `build-queue`, `review`, the vault). Jump straight to [Match your own catalog](#match-your-own-catalog-bring-your-own-marcxml) below. The mode-1 loop, *Daily flow B (label)*, and *Daily flow C (measure)* are for people growing and validating the shared training set; a matching-only user does not need them.

---

## Match your own catalog (bring your own MARCXML)

This is the most common external use: you have a MARC export from your own ILS and want to know which titles map to U.S. copyright registration & renewal records (the CCE). The result is a linkage table — every row is a *candidate* `(MARC record, CCE registration)` pair for a human to verify. The matcher **produces linkages; it does not decide copyright status.** A determination (registered, renewed, unclaimed, public domain) is reasoning *you* apply on top of the linkage, with the CCE evidence in hand.

Nothing here is institution-specific. The CCE side is the same for everyone; the only input that's yours is your MARCXML file. Three steps:

### 1. Build the CCE index (once)

The copyright-records side of the match is identical for every catalog, so you build it once and reuse it. It comes from the NYPL-transcribed CCE, which ships as **lazy** git submodules (~1.5 GB) that a plain clone skips — fetch them first, then build the LMDB index:

```bash
# Fetch the lazy NYPL CCE submodules (skipped by --recurse-submodules).
git submodule update --init --checkout data/nypl-reg data/nypl-ren

# Build the CCE LMDB index (~37 seconds). Registration XML + renewal TSV in,
# one queryable index out.
pdm run pd-matcher index build \
  --reg-dir data/nypl-reg/xml \
  --ren-dir data/nypl-ren/data \
  --out caches/cce.lmdb
```

You only repeat this when the NYPL submodules update or the parser changes (see [When to rebuild caches](#when-to-rebuild-caches)).

The score calibrators are local artifacts that live beside the index and are **not** checked in. On a fresh checkout (or after deleting `caches/`), regenerate them before matching, or the matchers run on raw, uncalibrated scores (a warning is logged when this happens):

```sh
pdm run fit-calibrator
pdm run python scripts/fit_learned_calibrator.py
```

### 2. Filter your MARCXML to in-scope records

`pd-groundtruth filter` takes a **local MARCXML file** and writes out only the records this matcher can do anything with — no download, no cap, no Princeton or any other institution involved:

```bash
pdm run pd-groundtruth filter \
  --input your-catalog.marcxml \
  --output in-scope.marcxml
```

A record is **in scope** when it is all of:

- a **monograph** in book format (leader byte 6 = `a`, byte 7 = `m`);
- **not an electronic resource** (no 007 electronic indicator, no online-resource carrier/extent, no "[electronic resource]" GMD);
- in a **supported language** — `eng`, `fre`, `ger`, `spa`, or `ita` (008 positions 35–37);
- published **within the moving wall through 1977** — from `today.year − 95` (the wall advances every January) up to and including 1977, the last year of CCE registrations under the 1909 Act;
- **not a government publication** (008 position 28); and
- **carrying a title** (245 subfield `a`).

Everything outside that window has no CCE registration to link to, so filtering first spares the matcher the large out-of-scope majority. (You *can* hand a raw file straight to `match --marc` — it accepts any MARCXML — but on a full catalog export, filtering first is much faster.) Pass `--languages eng,fre` to narrow further; the default keeps all five.

### 3. Match against the CCE index

```bash
pdm run pd-matcher match \
  --marc in-scope.marcxml \
  --index caches/cce.lmdb \
  --out matches.jsonl
```

Output is **JSONL** — one record per line. By default every input record gets a row, with blank `match_*` fields when nothing scored above the floor. Useful knobs:

- `--matches-only` — write rows only for genuinely matched pairs, skipping the no-match records.
- `--min-score N` — keep only pairs scoring ≥ `N` on the **0–100 calibrated scale** (overrides the config's `min_combined_score`, default `50`). `--min-score 90` is a strict triage cut.
- `--scorer weighted_mean|learned` — choose the combiner. The default `weighted_mean` is zero-dependency and works out of the box; `learned` is the LightGBM combiner, which needs the optional `ml` extra and a trained artifact (see [LEARNED_MATCHER.md](LEARNED_MATCHER.md)).
- `--workers N` — worker processes (default `cpu_count − 1`).

For a very large export, prepare it into re-runnable chunks first and match the chunk directory instead of the single file:

```bash
pdm run pd-matcher prepare-marc \
  --marc in-scope.marcxml \
  --out caches/prepared

pdm run pd-matcher match \
  --prepared caches/prepared \
  --index caches/cce.lmdb \
  --out matches.jsonl
```

Run `pdm run pd-matcher prepare-marc --help` for the chunking flags.

Each row in `matches.jsonl` is a candidate, not a verdict. Verifying which candidates are true links — and what each true link implies for copyright status — is the human step this tool exists to make possible. The labeling UI (*Daily flow B*) is one way to do that verification systematically, but it is not required to use the matcher.

### What's in each output row

Every row is a flat JSON object with the columns below (`output/jsonl_writer.py::RECORD_FIELDS`, the authoritative order). On a no-match record every `match_*` and score column is the empty string; the `marc_*` columns are always populated.

**Your MARC record.** Each text field appears three ways: `_original` (as catalogued), `_normalized` (diacritics stripped, punctuation collapsed), and `_stemmed` (the language-stemmed tokens the title scorer actually compares).

| column | meaning |
|---|---|
| `marc_id` | the MARC control identifier (001) |
| `marc_title_original` / `_normalized` / `_stemmed` | the fused title (245 `$a`+`$b`) |
| `marc_author_original` / `_normalized` / `_stemmed` | the statement of responsibility (245 `$c`) |
| `marc_main_author_original` / `_normalized` / `_stemmed` | the main author (1xx) |
| `marc_publisher_original` / `_normalized` / `_stemmed` | the publisher (264/260 `$b`) |
| `marc_year` | publication year |
| `marc_lccn` / `marc_lccn_normalized` | the LCCN as catalogued and canonicalized |
| `marc_country_code` / `marc_language_code` | 008 country and language codes |

**The matched CCE registration.** Empty on a no-match row.

| column | meaning |
|---|---|
| `match_type` | `registration` on a match, empty otherwise |
| `match_title` / `_normalized` | the CCE registration title |
| `match_author` / `_normalized` | the CCE author name |
| `match_publisher` / `_normalized` | the CCE publisher/claimant names, space-joined |
| `match_year` | the registration year |
| `match_source_id` | the CCE registration's NYPL uuid (the durable join key) |
| `match_date` | the registration date (ISO, or year-only when that's all that's known) |

**The copyright facts.** These carry the registration and renewal evidence a consumer reasons over. The renewal columns encode a specific provenance model — read the three notes below carefully.

| column | meaning |
|---|---|
| `match_regnum` | the matched registration's own CCE registration number |
| `match_prev_regnums` | prior registration numbers this record back-references (`<prev-regNum>`), semicolon-joined — the earlier or *ad interim* registrations of the same work |
| `match_was_renewed` | `true`/`false` — whether the matched registration's **own** renewal join fired. This reflects the record's own join only; it is **not** set from a sibling's renewal |
| `match_renewal_id` / `match_renewal_date` | the renewal's entry id and date. When the record was renewed these are its **own** renewal; when it was not, they carry a `<prev-regNum>`-linked sibling registration's renewal (propagated at index build) |
| `match_renewal_via` | **empty** when the renewal facts are the record's own; a **regnum** when the facts were inherited from that prev-regNum-linked sibling registration, naming which one |

The sibling case matters for *ad interim* books: the full registration may carry the renewal while the *ad interim* one does not (or vice versa). `match_renewal_via` tells you whether the renewal you see belongs to this exact registration or to its cross-linked sibling — see [docs/findings/recall_miss_forensics_2026-07-03.md](findings/recall_miss_forensics_2026-07-03.md) for why that distinction was added.

**The scores.**

| column | meaning |
|---|---|
| `title_score` / `author_score` / `publisher_score` | per-field scores, integer 0–100 (empty when the scorer was skipped) |
| `combined_score` | the combined confidence, `calibrated × 100` with two decimals — the value `--min-score` compares against |
| `year_difference` | `marc_year − match_year` |

### Confidence and calibration

`combined_score` is on a 0–100 scale, and `--min-score` filters against it. What that number *means* depends on whether a calibrator is present:

- **With `caches/calibrator.msgpack`** — `match` auto-loads it (from the index's parent directory) and the weighted-mean arm emits a **calibrated probability**: `--min-score 90` keeps only pairs at ≈ 90% confidence. The learned arm (`--scorer learned`) emits calibrated probabilities natively, so the threshold means the same thing for both combiners.
- **With no artifact** — the weighted arm **silently** falls back to `raw / 100` (uncalibrated), so `--min-score 90` is filtering on raw score, not 90% confidence, with nothing in the output to flag it. Making that non-silent is tracked at [issue #117](https://github.com/jpstroop/pd-matcher/issues/117).

The artifact is fit out-of-band from the labeled vault; nothing in `index` or `match` builds it. To fit or refit it:

```bash
pdm run python scripts/fit_calibrator.py
```

See [docs/findings/calibrator_refit_2026-07-03.md](findings/calibrator_refit_2026-07-03.md) for the current fit and [docs/DESIGN.md](DESIGN.md#platt-scaling-optional-off-by-default) for the math.

---

## The mode-1 (labeling) loop in 60 seconds

*This section and the two "Daily flow" sections after the setup are the **labeling** workflow — growing and validating the shared training set. If you came here only to match your own catalog, you do not need any of it; see [Match your own catalog](#match-your-own-catalog-bring-your-own-marcxml) above.*


```
NYPL CCE submodules                        Princeton MARC dump
         │                                          │
         ▼                                          ▼
[pdm run pd-matcher index build]   [pdm run pd-groundtruth acquire]
         │                                          │
         ▼                                          ▼
   caches/cce.lmdb                         data/candidates/
   (CCE index)                              (MARC pool)
         │                                          │
         └──────────────────┬───────────────────────┘
                            ▼
              [pdm run pd-groundtruth build-queue]
                            │
                            ▼
                   data/review.db   (stratified pairs to label)
                            │
                            ▼
                [pdm run pd-groundtruth review]   ← humans label here
                            │
                            ▼
              data/training/label_vault.jsonl   (durable; training submodule)
                            │
                            ▼
                  [pdm run pd-matcher eval]   ← measures matcher vs vault
```

Two persistent inputs (CCE submodules + Princeton MARC), two derived caches (LMDB index, MARC pool), one transient queue (review.db), one authoritative output (vault JSONL). The vault is the only file in the loop that's both human-produced and source-of-truth.

---

## Setup (once per machine)

Prereqs:

- macOS or Linux, recent shell
- [asdf](https://asdf-vm.com/) (recommended) for managing the Python version pin
- Standard CPython 3.14+ — **not** the free-threaded `t` build
- [PDM](https://pdm-project.org/)

One-time install from a fresh clone:

```bash
git clone --recurse-submodules <repo-url>
cd public_domain
pdm install
pdm run pre-commit install
```

That's it for code. The data comes in as git submodules. The labeled training bundle (vault + `marc.xml`) under `data/training/` is pulled normally by `--recurse-submodules`. The NYPL-transcribed CCE under `data/nypl-reg/` and `data/nypl-ren/` is **lazy** (~1.5 GB) — `--recurse-submodules` skips it, and so does a plain `git submodule update --init`. You fetch it on demand right before the first CCE index build (below). If you forgot `--recurse-submodules` entirely, `git submodule update --init` pulls `data/training`.

**The CCE index is required for everything** — both matching your own catalog and the labeling loop. Build it once (this is step 1 of [Match your own catalog](#match-your-own-catalog-bring-your-own-marcxml), repeated here for the setup checklist):

```bash
# 1. Fetch the lazy NYPL CCE submodules (skipped by --recurse-submodules; needed
#    before the first index build, and any time you don't yet have them locally).
git submodule update --init --checkout data/nypl-reg data/nypl-ren

# 2. Build the CCE LMDB index from the submodule data (~37 seconds).
pdm run pd-matcher index build \
  --reg-dir data/nypl-reg/xml \
  --ren-dir data/nypl-ren/data \
  --out caches/cce.lmdb
```

That index plus your own MARCXML is all you need to match — see [Match your own catalog](#match-your-own-catalog-bring-your-own-marcxml). **The rest of this Setup section is for the labeling loop only**; a matching-only user can stop here.

The labeling loop additionally needs a MARC pool to sample training pairs from. `acquire` downloads one (~1 hour first run); its default `--manifest-url` is currently stale (see the disk-guard note below), so pass a live full-dump manifest URL:

```bash
# Acquire a capped training-set MARC pool (labeling loop only).
pdm run pd-groundtruth acquire \
  --out-dir data/candidates \
  --manifest-url https://bibdata.princeton.edu/dumps/<live-dump>.json
```

Everything built here lands under `caches/` and `data/candidates/`, which are gitignored — derived, regenerable, and large.

`acquire` is the **capped training-set sampler**: it keeps at most `--per-decade-cap` records per (language, decade) bucket and writes per-language MARCXML shards under `data/candidates/<lang>/`, the pool the labeling loop samples from. It is *not* the whole-catalog extractor. Two sibling `pd-groundtruth` commands produce the full, uncapped in-scope corpus that production matching runs over:

- **`build-corpus`** — streams every dump in the manifest, keeps every in-scope (monograph, not an electronic resource, publication year within the moving wall .. 1977, supported language, has a title) record, and writes one MARCXML `<collection>` at `--output`. This is the whole-catalog extractor for `pd-matcher match` — never match the raw fixture or the capped pool.
- **`filter`** — the same uncapped in-scope extraction applied to a single local MARCXML file (`--input` → `--output`), for when you already have a file on disk. This is the on-ramp for the [Match your own catalog](#match-your-own-catalog-bring-your-own-marcxml) flow; it needs no manifest and no download.

```bash
# Whole-catalog in-scope corpus (uncapped):
pdm run pd-groundtruth build-corpus \
  --output data/corpus.marcxml \
  --manifest-url https://bibdata.princeton.edu/dumps/<live-dump>.json

# Or filter a local MARCXML file to its in-scope records:
pdm run pd-groundtruth filter \
  --input data/some-dump.marcxml \
  --output data/in-scope.marcxml
```

> **The default `--manifest-url` is stale.** The hardcoded default (`https://bibdata.princeton.edu/dumps/16368.json`) now returns a 404 page rather than a manifest, so `acquire` and `build-corpus` must be given a live full-dump manifest URL (a Princeton-bibdata-style dump-manifest endpoint) via `--manifest-url` until [issue #100](https://github.com/jpstroop/pd-matcher/issues/100) lands a current default.

`acquire` and `build-corpus` both stream large multi-dump downloads to disk. They abort safely (rather than filling the filesystem) if free space on either the temp-download directory or the output directory drops below `--min-free-space-mb` (default `2048` MB, i.e. 2 GB; the guard is on by default), checked before and during each download; pass `--min-free-space-mb 0` to disable it. On a shortfall they finalize the valid partial output (`build-corpus`'s partial `<collection>`, `acquire`'s partial per-language shards) and exit non-zero:

```bash
pdm run pd-groundtruth build-corpus \
  --output data/corpus.marcxml \
  --manifest-url https://bibdata.princeton.edu/dumps/<live-dump>.json \
  --min-free-space-mb 4096
```

---

## Daily flow A — operate the matcher

There are two ways to run the engine, one per mode (see *Two things this tool does* above):

- **`build-queue` (mode 1)** — match a sampled, stratified slice of the pool and write it to a SQLite labeling queue for the review UI.
- **`pd-matcher match` (mode 2)** — match MARC records and write a `(MARC, CCE)` linkage **JSONL** file (one record per line). This is the production matcher. Run it over the in-scope corpus from `build-corpus` (or `filter`), never the raw fixture or the capped pool.

```bash
# Produce labeled candidates for review, refreshing the queue from the pool.
pdm run pd-groundtruth build-queue --rebuild

# Or, for production matching of a prepared chunk directory:
pdm run pd-matcher match \
  --prepared data/prepared \
  --index caches/cce.lmdb \
  --out /tmp/matches.jsonl
```

(`pd-matcher match` takes either `--marc <single XML file>` or `--prepared <chunk dir produced by pd-matcher prepare-marc>`. Run `pdm run pd-matcher prepare-marc --help` for the chunking workflow.)

A few `match` knobs worth knowing:

- `--scorer weighted_mean|learned` overrides the combiner for the run. The default is `weighted_mean` (zero-dependency); `learned` selects the LightGBM combiner and needs the trained artifact plus the `ml` extra (see [LEARNED_MATCHER.md](LEARNED_MATCHER.md)).
- `--matches-only` writes only genuinely matched pairs. By default every input record gets one output row, with blank `match_*` fields when nothing scored above the floor.
- `--min-score` is on the **0–100 calibrated scale** (e.g. `--min-score 90` keeps only pairs scoring ≥ 90), overriding the config's `min_combined_score` (default `50`).

`build-queue` does the matching AND stratifies by language and confidence band, so you don't burn label effort on easy high-confidence pairs. Run `pdm run pd-groundtruth build-queue --help` for flags.

When tuning, the `--requeue VERDICT` flag (repeatable, valid values `match`/`no_match`/`unsure`) opts past vault verdicts back into the queue. The common case is `--requeue unsure` after a matcher improvement to re-look at previously ambiguous pairs.

---

## Daily flow B — label

```bash
pdm run pd-groundtruth review
```

Opens a local FastAPI server on port 8000 (default). The review card shows one pair at a time: MARC panel on the left, CCE panel on the right, evidence bars showing per-scorer confidence. Keyboard:

- `y` — match
- `n` — no_match
- `u` — unsure
- `s` or space — skip (no verdict recorded)
- `b` or ← — back to previous pair

The optional note field captures free text about anything notable. Notes accumulate and will be analyzed for patterns later.

Every verdict writes a line to `data/training/label_vault.jsonl` (in the training submodule) and a row to `data/review.db`'s `label` table. The vault is the source of truth; the DB is a transient working copy. Because the vault lives in the submodule, persisting it upstream means committing inside `data/training` and bumping the submodule pointer in the main repo (see *Publishing the training bundle*).

If you restart the server while developing, kill the process and re-run — uvicorn auto-reload isn't on; templates auto-reload but Python code changes require a restart.

See [docs/LABELING_GUIDE.md](LABELING_GUIDE.md) for what each verdict means and how to handle edge cases (translations, e-book reprints, near-duplicates).

---

## Daily flow C — measure

```bash
# Run the eval over the live vault.
pdm run pd-matcher eval \
  --vault data/training/label_vault.jsonl \
  --pool data/candidates \
  --index caches/cce.lmdb
```

Output: counts, per-MARC precision/recall, AUC, average precision, and a 21-point threshold sweep. The eval is read-only — it never modifies the vault or the index.

```bash
# Gate against the locked baseline (fails if P or R dropped > 2 pp).
pdm run regression

# Refresh the baseline after an intentional pipeline change.
pdm run regression-baseline
```

The regression gate is excluded from the default test suite (slow, index-dependent). Run it before merging changes that touch the matching pipeline.

---

## Maintenance

### Gates

Before any commit:

```bash
pdm run gates    # fmt + lint + typecheck + ~1000 unit tests at 100% coverage
pdm run webui    # the FastAPI integration suite (separate marker)
```

Gates failing is never acceptable: every commit must land with fmt, lint, typecheck, and the full test suite at 100% coverage all green. If a test becomes irrelevant, surface it for discussion and remove or rewrite it deliberately — never ignore, skip, or work around a failing test.

### When to rebuild caches

| Cache | Rebuild when |
|---|---|
| `caches/cce.lmdb` | NYPL submodules updated; parser/model changes |
| `data/candidates/` | Acquire-filter changes (e.g. e-book detection) |
| `data/review.db` | Pipeline changes that affect scoring or banding (`build-queue --rebuild`) |

The vault never gets rebuilt — it's append-only and survives all of the above.

### Vault schema migrations

The vault carries a `schema` integer per line. When that bumps, run the corresponding CLI subcommand:

```bash
pdm run pd-groundtruth migrate-vault-v5   # most recent (categories backfill)
```

Migrations are idempotent and write atomically — re-running a migration that's already done is a logged no-op.

### Publishing the training bundle

The training data lives in a **submodule** at `data/training/` — the [`jpstroop/cce-marc-linkage`](https://github.com/jpstroop/cce-marc-linkage) repo, pinned by the main repo. It holds exactly two files:

- `label_vault.jsonl` — the vault itself. The labeling UI writes verdicts straight here, so it is always current. This is the source of truth *and* the training labels (the full record, including the labeler's notes).
- `marc.xml` — MARCXML of every MARC the vault references (regenerated by `dump-vault-marcs`), so the pairs can be re-scored without the full candidate pool.

There is no separate reshape step and no `matches.jsonl`/`training.jsonl`: a frozen matches list is only valid for one catalog, and the vault *is* the training table. Because the vault lives in the submodule, publishing is just ordinary submodule hygiene:

```bash
# Refresh the MARC snapshot to match the current vault.
pdm run pd-groundtruth dump-vault-marcs        # writes data/training/marc.xml

# Commit + push inside the submodule (the vault is already current there,
# written by the labeling UI):
cd data/training
git add label_vault.jsonl marc.xml
git commit -m "regenerate from vault @ N entries"
git push origin main
cd -

# Record the new submodule commit in the main repo:
git add data/training
git commit -m "bump training submodule"
```

`dump-vault-marcs` reads the vault and `data/candidates/`, walks shards streamingly, and writes a single MARCXML file (default `data/training/marc.xml`). It reports `vault_entries`, `distinct_marcs_requested`, `marcs_written`, and `marcs_missing` — the missing count is vault entries whose MARC no longer exists in the candidate pool. It is read-only against the vault; safe to run anytime, including mid-labeling-session.

To **train the learned matcher** from this bundle, see [LEARNED_MATCHER.md](LEARNED_MATCHER.md).

### Regression baseline

`tests/regression/baseline.json` is the locked snapshot of what the matcher's accuracy looked like at the time of the last intentional change. Two commands:

- `pdm run regression-baseline` — measures the current matcher against the current vault and overwrites `baseline.json`. Run this after a pipeline change you *intended* to make.
- `pdm run regression` — runs the eval and compares against the locked baseline. Fails if precision OR recall dropped more than 2 percentage points. AUC/AP are reported but not yet gated.

---

## Where things live

| Path | Contents | Tracked? |
|---|---|---|
| `src/pd_matcher/` | Core matching library | yes |
| `src/pd_groundtruth/` | Labeling subsystem (CLI + FastAPI review) | yes |
| `tests/` | unit + groundtruth + integration + regression | yes |
| `data/nypl-reg/`, `data/nypl-ren/` | NYPL-transcribed CCE | submodule |
| `data/training/` | Training-bundle submodule ([`cce-marc-linkage`](https://github.com/jpstroop/cce-marc-linkage)): the vault `label_vault.jsonl` (source of truth for labels) + `marc.xml` (target of `dump-vault-marcs`) | submodule |
| `data/candidates/` | MARC pool (acquire output; your own data) | no (gitignored) |
| `data/*.db` | Transient review queues (SQLite) | no (gitignored) |
| `caches/cce.lmdb/` | Built CCE index | no |
| `logs/` | Per-run log files | no |
| `docs/` | Deep-dive docs (design, glossary, architecture, this guide) | yes |
| `tests/regression/baseline.json` | Locked accuracy snapshot | yes |

---

## When something breaks

- **"Module not found" on a fresh shell** → activate via `pdm run …`. Never call `python`, `pytest`, `mypy`, etc. directly.
- **Review UI changes don't appear** → restart the `pd-groundtruth review` process. Template edits auto-reload; Python code edits don't.
- **Vault decode errors** → check the `schema` field on the offending line. If it's lower than current `SCHEMA_VERSION`, run the matching `migrate-vault-vN` CLI.
- **`pdm run regression` fails after an intentional change** → that's the gate working. If the change was wanted, `pdm run regression-baseline` to refresh the lock, then commit the new `baseline.json`.
- **Eval reports many `marc_not_in_pool` warnings** → the pool was rebuilt with a different filter; old vault entries lost their MARCs. The eval drops them gracefully; if the drop count is large, consider why the pool shrunk.
- **`pdm install` fails on Python version** → check `.tool-versions`; asdf should pick up the right CPython. **Never use the free-threaded `t` build** — strict, no exceptions.

---

## Further reading

- [README.md](../README.md) — what + why; one-screen overview for new collaborators and stakeholders.
- [docs/DESIGN.md](DESIGN.md) — the matching algorithm, end to end: parsing, normalization, indexing, scoring, calibration.
- [docs/MATCHING_ARCHITECTURE.md](MATCHING_ARCHITECTURE.md) — candidate retrieval vs scoring as separate concerns.
- [docs/GLOSSARY.md](GLOSSARY.md) — plain-language definitions of every domain term.
- [docs/LABELING_WORKFLOW.md](LABELING_WORKFLOW.md) — the labeler's operational playbook: every command in order with trigger conditions for queue rebuild, publishing, and the diagnostic.
- [docs/LABELING_GUIDE.md](LABELING_GUIDE.md) — the labeler's decision guide for verdicts and edge cases.
- [docs/studies/](studies/) — write-ups of one-off measurement runs (year-window study, field-pairing experiments, etc.).
- [docs/LEARNED_MATCHER.md](LEARNED_MATCHER.md) — the production learned (LightGBM) combiner: what it is, how to train it (`train-scorer`), how to train from the `data/training` bundle, and why both it and the weighted mean exist.
- [docs/LEARNED_SCORER_DIAGNOSTIC.md](LEARNED_SCORER_DIAGNOSTIC.md) — the original read-only diagnostic that preceded the production combiner (historical; see LEARNED_MATCHER.md for the current model).
- GitHub issues at <https://github.com/jpstroop/pd-matcher/issues> — active work and tracked decisions.
