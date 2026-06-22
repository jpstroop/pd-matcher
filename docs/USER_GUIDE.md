# User guide

A guided tour of the operational picture: mental model, daily flows, maintenance. If you're looking for *what* this project is and *why* it exists, start with the [README](../README.md) — this guide picks up after you've decided to actually run it. For the matching algorithm itself, see [docs/DESIGN.md](DESIGN.md).

---

## Two things this tool does

Two distinct modes live in this repo, and most confusion comes from blurring them:

1. **Build & review a training set** — *the browser-UI workflow.* You sample candidate `(MARC, CCE)` pairs, judge each one in a local review UI, and accumulate a verified vault. That vault is the ground truth that trains and measures the matchers, and it is the bulk of day-to-day work here. Commands: `pd-matcher index build` plus the `pd-groundtruth` family (`acquire`, `build-queue`, `review`).

2. **Match a catalog to produce pairs** — *the matcher as a tool.* You run the matcher over MARC records to emit a `(MARC record, CCE registration)` linkage table — the actual "which of my books appear in these copyright records" use. Command: `pd-matcher match` (→ a CSV).

Both modes share one engine — the CCE index plus the per-field scorers — and differ only in what they emit: mode 1 builds a labeling queue and grows the vault; mode 2 writes a linkage table. The matcher that powers mode 2 is exactly what mode 1's vault trains and validates. The diagram below traces **mode 1**; mode 2 is a single `pd-matcher match` run (see *Daily flow A*).

---

## The mode-1 loop in 60 seconds

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

Two data caches need to be built before the matcher can do anything useful. Order matters:

```bash
# 1. Fetch the lazy NYPL CCE submodules (skipped by --recurse-submodules; needed
#    before the first index build, and any time you don't yet have them locally).
git submodule update --init --checkout data/nypl-reg data/nypl-ren

# 2. Build the CCE LMDB index from the submodule data (~37 seconds).
pdm run pd-matcher index build \
  --reg-dir data/nypl-reg/xml \
  --ren-dir data/nypl-ren/data \
  --out caches/cce.lmdb

# 3. Acquire and filter a MARC pool from Princeton (~1 hour first run).
pdm run pd-groundtruth acquire --out-dir data/candidates
```

Both produce files under `caches/` and `data/candidates/` that are gitignored — they're derived, regenerable, and large.

`acquire` and `build-corpus` both stream large multi-dump downloads to disk. They abort safely (rather than filling the filesystem) if free space on either the temp-download directory or the output directory drops below `--min-free-space-mb` (default `2048`), checked before and during each download; pass `--min-free-space-mb 0` to disable the guard. When `build-corpus` aborts this way it still finalizes a valid partial `<collection>` and exits non-zero:

```bash
pdm run pd-groundtruth build-corpus --output data/corpus.marcxml --min-free-space-mb 4096
```

---

## Daily flow A — operate the matcher

There are two ways to run the engine, one per mode (see *Two things this tool does* above):

- **`build-queue` (mode 1)** — match a sampled, stratified slice of the pool and write it to a SQLite labeling queue for the review UI.
- **`pd-matcher match` (mode 2)** — match MARC records and write a `(MARC, CCE)` linkage **CSV**. This is the production matcher.

```bash
# Produce labeled candidates for review, refreshing the queue from the pool.
pdm run pd-groundtruth build-queue --rebuild

# Or, for production matching of a prepared chunk directory:
pdm run pd-matcher match \
  --prepared data/prepared \
  --index caches/cce.lmdb \
  --out /tmp/matches.csv
```

(`pd-matcher match` takes either `--marc <single XML file>` or `--prepared <chunk dir produced by pd-matcher prepare-marc>`. Run `pdm run pd-matcher prepare-marc --help` for the chunking workflow.)

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

Gates failing is never acceptable — see the standing rules in the repo's coding standards. If a test becomes irrelevant, surface that and discuss; don't ignore it.

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
