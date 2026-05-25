# User guide

A guided tour. If you're returning to this project after time away, or
onboarding for the first time, read this top-to-bottom — it's designed
to take ten minutes and leave you knowing where everything is. For
per-command reference, see [README.md](../README.md). For the matching
algorithm itself, see [docs/design.md](design.md).

---

## What this is

`pd-matcher` produces a verified linkage table between Princeton's
MARC catalog and the NYPL transcription of the U.S. Copyright Office's
Catalog of Copyright Entries (CCE). One row of output = one
`(MARC record, CCE registration, optional CCE renewal)` triple, with
per-field scores and a calibrated confidence.

It does **not** decide public-domain status. Consumers apply their own
copyright reasoning to the linkage. Treating this as a PD-determination
tool will lead you astray.

The companion `pd-groundtruth` CLI is how humans build the labeled
corpus the matcher's calibration and evaluation depend on.

---

## The mental model in 60 seconds

```
NYPL CCE submodules                   Princeton MARC dump
        │                                     │
        ▼                                     ▼
[pd-matcher index build]              [pd-groundtruth acquire]
        │                                     │
        ▼                                     ▼
caches/nypl.lmdb                      data/candidates/  (MARC pool)
   (CCE index)                                │
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
              [pd-groundtruth build-queue]
                       │
                       ▼
                data/review.db   (stratified pairs to label)
                       │
                       ▼
              [pd-groundtruth review]  ← humans label here
                       │
                       ▼
              data/label_vault.jsonl   (durable, git-tracked)
                       │
                       ▼
              [pd-matcher eval]   ← measures matcher vs vault
```

Two persistent inputs (CCE submodules + Princeton MARC), two derived
caches (LMDB index, MARC pool), one transient queue (review.db), one
authoritative output (vault JSONL). The vault is the only file in the
loop that's both human-produced and source-of-truth.

---

## Roles

You'll wear one or more of these hats:

- **Operator** — runs the matcher pipeline: acquire fresh MARC, build
  the index, produce candidate pairs. Mostly batch jobs.
- **Labeler** — opens the review UI, looks at proposed pairs, clicks
  match / no_match / unsure with optional notes. Produces the vault.
- **Maintainer** — keeps the code passing gates, refreshes the
  regression baseline after intentional pipeline changes, runs
  migrations when the vault schema bumps.

All three are local-machine roles today. The GCP deployment plan
(GitHub #34) is the path to multi-user labeling behind Google OAuth.

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

That's it for code. The CCE registration and renewal data come in as
git submodules under `data/nypl-reg/` and `data/nypl-ren/`. If you
forgot `--recurse-submodules`, run `git submodule update --init`.

Two data caches need to be built before the matcher can do anything
useful. Order matters:

```bash
# 1. Build the CCE LMDB index from the submodule data (~37 seconds).
pdm run pd-matcher index build \
  --reg-dir data/nypl-reg/xml \
  --ren-dir data/nypl-ren/data \
  --out caches/nypl.lmdb

# 2. Acquire and filter a MARC pool from Princeton (~1 hour first run).
pdm run pd-groundtruth acquire
```

Both produce files under `caches/` and `data/candidates/` that are
gitignored — they're derived, regenerable, and large.

---

## Daily flow A — operate the matcher

You're producing match candidates against the index. This is what
`pd-matcher match` is for in production; `build-queue` is the labeling
equivalent that also stratifies the result into a SQLite queue.

```bash
# Produce labeled candidates for review, refreshing the queue from the pool.
pdm run pd-groundtruth build-queue --rebuild

# Or, for production matching of a prepared chunk directory:
pdm run pd-matcher match \
  --prepared data/prepared \
  --index caches/nypl.lmdb \
  --out /tmp/matches.csv
```

(`pd-matcher match` takes either `--marc <single XML file>` or
`--prepared <chunk dir produced by prepare-marc>`. See the
[`prepare-marc` README section](../README.md#how-it-fits-together) for
the chunking workflow.)

`build-queue` does the matching AND stratifies by language and
confidence band, so you don't burn label effort on easy high-confidence
pairs. See the [README's `build-queue` section](../README.md#2-build-queue--match-and-stratify-into-a-review-queue) for flags.

When tuning, the `--requeue VERDICT` flag (repeatable, valid values
`match`/`no_match`/`unsure`) opts past vault verdicts back into the
queue. The common case is `--requeue unsure` after a matcher
improvement to re-look at previously ambiguous pairs.

---

## Daily flow B — label

```bash
pdm run pd-groundtruth review
```

Opens a local FastAPI server on port 8000 (default). The review card
shows one pair at a time: MARC panel on the left, CCE panel on the
right, evidence bars showing per-scorer confidence. Keyboard:

- `y` — match
- `n` — no_match
- `u` — unsure
- `s` or space — skip (no verdict recorded)
- `b` or ← — back to previous pair

The optional note field captures free text about anything notable.
Notes accumulate and will be analyzed for patterns later.

Every verdict writes a line to `data/label_vault.jsonl` and a row to
`data/review.db`'s `label` table. The vault is the source of truth; the
DB is a transient working copy.

If you restart the server while developing, kill the process and
re-run — uvicorn auto-reload isn't on; templates auto-reload but
Python code changes require a restart.

See [docs/LABELING_GUIDE.md](LABELING_GUIDE.md) for what each verdict
means and how to handle edge cases (translations, e-book reprints,
near-duplicates).

---

## Daily flow C — measure

```bash
# Run the eval over the live vault.
pdm run pd-matcher eval \
  --vault data/label_vault.jsonl \
  --pool data/candidates \
  --index caches/nypl.lmdb
```

Output: counts, per-MARC precision/recall, AUC, average precision, and
a 21-point threshold sweep. The eval is read-only — it never modifies
the vault or the index.

```bash
# Gate against the locked baseline (fails if P or R dropped > 2 pp).
pdm run regression

# Refresh the baseline after an intentional pipeline change.
pdm run regression-baseline
```

The regression gate is excluded from the default test suite (slow,
index-dependent). Run it before merging changes that touch the
matching pipeline.

---

## Maintenance

### Gates

Before any commit:

```bash
pdm run gates    # fmt + lint + typecheck + ~1000 unit tests at 100% coverage
pdm run webui    # the FastAPI integration suite (separate marker)
```

Gates failing is never acceptable — see the standing rules in the
repo's coding standards. If a test becomes irrelevant, surface that and
discuss; don't ignore it.

### When to rebuild caches

| Cache | Rebuild when |
|---|---|
| `caches/nypl.lmdb` | NYPL submodules updated; parser/model changes |
| `data/candidates/` | Acquire-filter changes (e.g. e-book detection) |
| `data/review.db` | Pipeline changes that affect scoring or banding (`build-queue --rebuild`) |

The vault never gets rebuilt — it's append-only and survives all of
the above.

### Vault schema migrations

The vault carries a `schema` integer per line. When that bumps, run
the corresponding CLI subcommand:

```bash
pdm run pd-groundtruth migrate-vault-v4   # most recent (CCE IDs)
```

Migrations are idempotent and write atomically — re-running a migration
that's already done is a logged no-op.

### Regression baseline

`tests/regression/baseline.json` is the locked snapshot of what the
matcher's accuracy looked like at the time of the last intentional
change. Two commands:

- `pdm run regression-baseline` — measures the current matcher against
  the current vault and overwrites `baseline.json`. Run this after a
  pipeline change you *intended* to make.
- `pdm run regression` — runs the eval and compares against the locked
  baseline. Fails if precision OR recall dropped more than 2 percentage
  points. AUC/AP are reported but not yet gated.

---

## Where things live

| Path | Contents | Tracked? |
|---|---|---|
| `src/pd_matcher/` | Core matching library | yes |
| `src/pd_groundtruth/` | Labeling subsystem (CLI + FastAPI review) | yes |
| `tests/` | unit + groundtruth + integration + regression | yes |
| `data/nypl-reg/`, `data/nypl-ren/` | CCE submodules | submodule |
| `data/label_vault.jsonl` | Vault (source of truth for labels) | **yes** |
| `data/candidates/` | MARC pool (acquire output) | no (gitignored) |
| `data/review.db` | Transient label queue (SQLite) | no |
| `caches/nypl.lmdb/` | Built CCE index | no |
| `logs/` | Per-run log files | no |
| `docs/` | Deep-dive docs (design, glossary, architecture, this guide) | yes |
| `tests/regression/baseline.json` | Locked accuracy snapshot | yes |

---

## When something breaks

- **"Module not found" on a fresh shell** → activate via `pdm run …`.
  Never call `python`, `pytest`, `mypy`, etc. directly.
- **Review UI changes don't appear** → restart the `pd-groundtruth
  review` process. Template edits auto-reload; Python code edits don't.
- **Vault decode errors** → check the `schema` field on the offending
  line. If it's lower than current `SCHEMA_VERSION`, run the matching
  `migrate-vault-vN` CLI.
- **`pdm run regression` fails after an intentional change** → that's
  the gate working. If the change was wanted, `pdm run
  regression-baseline` to refresh the lock, then commit the new
  `baseline.json`.
- **Eval reports many `marc_not_in_pool` warnings** → the pool was
  rebuilt with a different filter; old vault entries lost their MARCs.
  The eval drops them gracefully; if the drop count is large, consider
  why the pool shrunk.
- **`pdm install` fails on Python version** → check `.tool-versions`;
  asdf should pick up the right CPython. **Never use the
  free-threaded `t` build** — strict, no exceptions.

---

## Further reading

- [README.md](../README.md) — per-command reference, output schema, all
  CLI flags.
- [docs/design.md](design.md) — the matching algorithm, end to end:
  parsing, normalization, indexing, scoring, calibration.
- [docs/matching-architecture.md](matching-architecture.md) —
  candidate retrieval vs scoring as separate concerns.
- [docs/glossary.md](glossary.md) — plain-language definitions of
  every domain term.
- [docs/LABELING_GUIDE.md](LABELING_GUIDE.md) — the labeler's decision
  guide for verdicts and edge cases.
- [docs/studies/](studies/) — write-ups of one-off measurement runs
  (year-window study, field-pairing experiments, etc.).
- GitHub issues at <https://github.com/jpstroop/pd-matcher/issues> —
  active work and tracked decisions.
