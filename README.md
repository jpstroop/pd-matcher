# pd-matcher

A command-line tool that scans large MARC XML catalogs against the U.S. Copyright Office's Catalog of Copyright Entries (CCE) — published by the Library of Congress and transcribed into XML/TSV by NYPL — and assigns each MARC record a public-domain status with a calibrated confidence score.

Matches are produced from a pipeline of small, pure-function **scorers** that each return structured **Evidence**, combined into a weighted-mean score, then mapped to a probability by a **Platt-scaled calibrator** trained against the project's 19,970-row ground-truth set. A separate rule engine evaluates the **Cornell public-domain decision matrix** (Categories 2 and 3, the ones that apply to books) and produces a copyright status with a human-readable explanation of any pragmatic assumptions used.

Output is a streaming CSV that mirrors the ground-truth schema, suitable for direct comparison and downstream analysis.

---

## Quick start

```bash
# Install (once per clone)
pdm install
pdm run pre-commit install

# Build the CCE index (one-time, ~37 s for the full corpus)
pdm run pd-matcher index build \
  --reg-dir data/nypl-reg/xml \
  --ren-dir data/nypl-ren/data \
  --out caches/cce.lmdb

# Inspect the built index
pdm run pd-matcher index info --lmdb-path caches/cce.lmdb

# Match a MARC file against the index
pdm run pd-matcher match \
  --marc data/candidate_marc_file.marcxml \
  --index caches/cce.lmdb \
  --out results.csv

# Evaluate against the ground-truth pairs
pdm run pd-matcher eval \
  --ground-truth data/combined_ground_truth.csv \
  --index caches/cce.lmdb \
  --report eval.json
```

The first time `pd-matcher match` runs against an index, it builds (and caches) a token-IDF table from the CCE titles. Subsequent runs reuse the cache.

---

## Installation

**Requirements:** Python 3.14+ (standard CPython — *not* the free-threaded `t` build), [PDM](https://pdm-project.org/), [asdf](https://asdf-vm.com/) recommended.

```bash
git clone --recurse-submodules <repo-url>
cd public_domain
pdm install
pdm run pre-commit install
```

The CCE registration and renewal data is pulled in via two git submodules:

- `data/nypl-reg/` — NYPL's transcription of the CCE registrations (1923–1977), ~2.17M records.
- `data/nypl-ren/` — NYPL's transcription of the CCE renewals (1950–2001), ~444k records, two distinct header schemas (pre-1978 vs `*-from-db.tsv`) handled transparently by the parser.

If you forgot `--recurse-submodules`, run `git submodule update --init`.

---

## Commands

### `pd-matcher index build`

Build the LMDB-backed CCE index. Streams the registration XML and renewal TSV files, normalizes each record, precomputes the registration↔renewal join (so workers never join at match time), and writes one mmap'd LMDB environment.

```bash
pd-matcher index build \
  --reg-dir data/nypl-reg/xml \
  --ren-dir data/nypl-ren/data \
  --out caches/cce.lmdb \
  [--force]
```

- `--force` rebuilds even when the source-hash + schema-version in the existing index match the inputs. Without it, an unchanged index is a no-op.

The index is a directory containing LMDB data files. Memory map size defaults to 16 GiB virtual; on-disk size for the current corpus is roughly 1.5 GB.

### `pd-matcher index info`

Print build statistics for an existing index — counts, source hash, build timestamp.

```bash
pd-matcher index info --lmdb-path caches/cce.lmdb
```

### `pd-matcher match`

Stream a MARC XML file through the matcher and write results to CSV.

```bash
pd-matcher match \
  --marc data/candidate_marc_file.marcxml \
  --index caches/cce.lmdb \
  --out results.csv \
  [--workers N] \
  [--year-window N] \
  [--min-score F] \
  [--as-of YYYY]
```

- `--workers N` — number of worker processes. Defaults to `cpu_count - 1`. On a 32-core box, that's 31 matching processes plus one for the writer plus the producer and reporter in main.
- `--year-window N` — overrides the matching config's year window (default 2). The matcher only considers CCE records published within `±N` years of the MARC record.
- `--min-score F` — overrides the matching config's minimum calibrated score (default 0.70). Below this threshold a candidate is dropped from the result.
- `--as-of YYYY` — reference *year* for the moving wall, defaults to the current year. Use this to reproduce historical runs or to test what enters PD on Jan 1 of a future year. Accepted range is 1923–2100.

The match command streams: it never holds the whole MARC file in memory. Workers share the LMDB index via the OS page cache, so memory overhead grows with worker count only by a small fixed amount per process.

`Ctrl-C` triggers a graceful shutdown: workers finish their current record, the queue drains, the CSV file is flushed and closed cleanly, and a partial-results path is printed. A second `Ctrl-C` aborts hard.

### `pd-matcher eval`

Run the matcher against the project's ground-truth set and produce precision / recall / F1 plus a per-status confusion matrix.

```bash
pd-matcher eval \
  --ground-truth data/combined_ground_truth.csv \
  --index caches/cce.lmdb \
  [--report eval.json] \
  [--limit N] \
  [--as-of YYYY]
```

- `--report PATH` — write the full `EvalReport` as JSON.
- `--limit N` — evaluate only the first `N` rows; useful for fast smoke testing.

The eval reconstructs a `MarcRecord` from each ground-truth row, runs the full match + assessment pipeline, and compares the predicted best match's `match_source_id` against the ground-truth `match_source_id`.

### `pd-matcher train-scorer`

Phase 9 placeholder. Will eventually train a LightGBM model on the per-Evidence feature vectors and persist it alongside the index. Currently exits 2 with a "not yet implemented" message.

### Global flags

- `--log-level DEBUG|INFO|WARNING|ERROR` (default INFO).
- `--json-logs` — emit structured JSON logs instead of human-readable lines. Useful for piping to log aggregators.
- `--quiet` — suppress everything below WARNING; overrides `--log-level`.

---

## Output format

The CSV mirrors `data/combined_ground_truth.csv` column-for-column, so any tool that consumes ground truth can consume `pd-matcher match` output:

| Column | Description |
|---|---|
| `marc_id` | MARC `001` control number |
| `marc_title_original` / `_normalized` / `_stemmed` | MARC `245$a $b`, raw / NFKD-lowercased / Snowball-stemmed |
| `marc_author_original` / `_normalized` / `_stemmed` | MARC `245$c` (statement of responsibility) |
| `marc_main_author_original` / `_normalized` / `_stemmed` | MARC `100/110/111$a` |
| `marc_publisher_original` / `_normalized` / `_stemmed` | MARC `260/264$b` |
| `marc_year` | parsed from `260/264$c` or `008` positions 7–10 |
| `marc_lccn` / `_normalized` | MARC `010$a` |
| `marc_country_code` / `marc_language_code` | MARC `008` positions 15–17 / 35–37 |
| `match_type` | currently always `registration` (renewal-only matches are a future extension) |
| `match_title` / `_normalized` | matched CCE record's title |
| `match_author` / `_normalized` | matched CCE record's authorName |
| `match_publisher` / `_normalized` | joined CCE publisher names |
| `match_year` | CCE registration year |
| `match_source_id` | NYPL UUID of the matched CCE record (stable identifier) |
| `match_date` | CCE registration date (ISO) |
| `title_score` / `author_score` / `publisher_score` | per-field Evidence score, 0–100 |
| `combined_score` | calibrated probability × 100 — *not* a raw weighted mean |
| `year_difference` | signed (`marc_year - match_year`) |
| `copyright_status` | one of 16 enum values from `CopyrightStatus` |

When no candidate clears the threshold, the `match_*` columns are blank and `copyright_status` is typically `UNKNOWN_INSUFFICIENT_DATA` or, for older works, `PD_BY_AGE_PRE_95_YEARS`.

---

## How it works

Matching MARC records to CCE registrations is hard for non-obvious reasons. ISBNs barely existed during the CCE period. LCCNs are present in some MARC records but absent from the CCE side. Titles and authors drift between sources (transcription errors, abbreviation differences, embedded edition info, language conventions). The corpus is multilingual. Years drift by 1–2 between publication, registration, and renewal. A naïve "for every MARC record, score it against every CCE record" pass would be ~2.17M × millions = trillions of comparisons.

The tool solves this in five layers:

**1. Blocking by year.** Every CCE record is indexed into a year bucket. For a MARC record published in 1955, the matcher only retrieves candidates with `reg_year ∈ [1953, 1957]`. The year window is configurable; default ±2.

**2. Per-field scoring.** Each candidate is scored against the MARC record by a set of pure-function scorers — one per signal (title, author, publisher, year, LCCN, ISBN, edition). Each scorer emits a structured `Evidence` object containing its score and named sub-features. Skipped scorers (missing fields) contribute nothing rather than penalizing.

**3. Field-pair permutations.** For known-confused field pairs (`marc.title` vs `nypl.series_titles`, `marc.publisher` vs `nypl.claimants`), the matcher runs both pairings and keeps the highest-scoring Evidence per scorer. The runners-up are preserved for audit.

**4. Combination + calibration.** A weighted-mean combiner reduces the Evidence collection to a single raw score. A Platt-scaled logistic regression, trained against the project's 19,970-row ground-truth set, maps the raw score to a calibrated probability. A "75" in the published `combined_score` column means there's roughly a 75% chance the pair is a true match, not "I gave this 75 vibe points."

**5. Copyright assessment.** The matched record (plus the MARC record itself) is handed to a rule engine that codifies Cornell's public-domain decision matrix. A moving-wall short-circuit handles the easy case (anything more than 95 years old is PD by age). Otherwise the engine walks ordered rules covering Cornell's Category 2 (US-registered or US-published) and Category 3 (foreign-published) and returns a `CopyrightAssessment` with the matched rule name, the leaf status, and any pragmatic assumptions used.

The matching pipeline is parallelized via Python's `multiprocessing` with the `spawn` start method. Workers share the LMDB index through the OS page cache (memory-mapped reads, zero-copy across processes). The producer streams MARC records from disk, batches them, and feeds workers via a bounded queue (backpressure). A single writer process consumes results and serializes the CSV. A reporter thread aggregates throughput, ETA, and per-status counts.

For a deep dive into the algorithm, scoring math, and design rationale, see [DESIGN.md](DESIGN.md).

---

## Performance

On a 32-core machine with the full submodule corpus:

- **Index build**: ~37 seconds.
- **Match throughput**: roughly 1–3k MARC records per second per worker depending on field complexity and number of year-bucket candidates per record. With 31 workers, a million-record MARC file finishes in tens of seconds, dominated by the LMDB read and the rapidfuzz scorers.
- **Memory**: each worker process is small (~50 MB Python overhead + ~10 MB scorer state); the LMDB file is mapped once and shared via the kernel page cache.

The `slow` pytest marker excludes one full-corpus integration test from the default test run; invoke it with `pdm run pytest -m slow` to verify end-to-end behavior against the real submodules.

---

## Configuration

The matcher and the rule engine each ship with shippable-default YAML files:

- `src/pd_matcher/config/defaults/matching.yaml` — scorer weights, year window, min combined score, scorer selection.
- `src/pd_matcher/config/defaults/copyright_rules.yaml` — ordered Cornell rules with predicate calls, status mappings, and assumption notes.

Both load through `src/pd_matcher/config/loader.py` against `msgspec.Struct` schemas in `src/pd_matcher/config/schemas.py`. Schema-violating YAML fails loudly at load time. Future work will let a user-provided YAML override or merge with the defaults; for now, edit the defaults directly if you need to tune.

---

## Development

```bash
pdm run gates          # mypy strict + ruff check + ruff format check + pytest with 100% line+branch coverage
pdm run fmt            # ruff format
pdm run lint           # ruff check
pdm run lint-fix       # ruff check --fix (autofix where safe)
pdm run typecheck      # mypy
pdm run test           # pytest (excludes the slow integration test by default)
pdm run pytest -m slow # run only the slow tests
```

The pre-commit hook runs `ruff format`, `ruff check --fix`, end-of-file fixer, trailing-whitespace fixer, and merge-conflict detection on every `git commit`. The heavier gates (mypy, pytest) are deliberately *not* in the pre-commit chain — they run via `pdm run gates` and (eventually) CI.

Project standards:

- Strict mypy with `disallow_any_explicit = true`. No `Any`. No `# type: ignore`.
- One import per line (`from os import getpid`, not `import os` and not `from os import getpid, unlink`).
- One class per file (private helpers may colocate).
- msgspec `Struct` for typed records throughout.
- 100% line + branch test coverage enforced.

See [DESIGN.md](DESIGN.md) for the technology decisions and what motivates them.
