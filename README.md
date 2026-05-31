# pd-matcher

A command-line tool that links Princeton's MARC catalog records to the U.S. Copyright Office's Catalog of Copyright Entries (CCE) — published by the Library of Congress and transcribed into XML/TSV by NYPL. Each MARC record is paired with its best-matching CCE registration (and any matching renewal) with a calibrated confidence score. Ships with the `pd-groundtruth` companion CLI used to build the human-labeled ground-truth corpus the matcher is calibrated against.

Matches are produced from a pipeline of small, pure-function **scorers** that each return structured **Evidence**, combined into a weighted-mean score, then mapped to a probability by a **Platt-scaled calibrator** trained against the project's ground-truth set.

Output is a streaming CSV: one row per MARC record, with the matched CCE registration's id, year, and per-field scores. Anyone who needs copyright status from this dataset can apply their own analysis to the verified linkage; this project does not make copyright determinations.

> **New here, or returning after a break?** Read [docs/USER_GUIDE.md](docs/USER_GUIDE.md) first — it's a 10-minute guided tour of setup, daily workflows, and maintenance. The rest of this README is per-command reference.

---

## Background

`pd-matcher` produces a **verified linkage** between Princeton's MARC bibliographic records and the **U.S. Copyright Office's Catalog of Copyright Entries (CCE)**: for each MARC record published in the window the CCE covers (1891–1977), the matcher finds the best-matching CCE registration (and any matching renewal) and emits one row per pair.

The published artifact is the linkage table — `(MARC record, CCE registration, optional CCE renewal)` triples — not a public-domain list. Consumers who need PD status can apply whatever copyright reasoning they want to the linkage: Cornell's decision matrix, the URAA restoration rules, country-of-origin analysis, etc. That analysis belongs to the consumer, not to this project.

The matcher's pipeline is fuzzy by necessity — titles get transcribed differently across the two corpora, authors get truncated, OCR garbles characters, and the CCE's renewal records are partial. To know how *good* the matcher's linkage calls are (and to improve them), we need a labeled corpus: pairs of `(MARC record, CCE registration)` where a human has confirmed **match**, **no_match**, or **unsure**. That labeled corpus is the **ground truth**, and producing it is what the `pd-groundtruth` half of this project exists to do.

The one copyright-aware decision the project makes is at acquire time: records published ≤ `today.year − 95` are already PD by age regardless of registration/renewal status, so linking them carries no signal and they are filtered out before matching. Everything past that point is pure linkage.

Who reads the labels:

- **The matcher itself** — for calibrating score thresholds (where does confidence become reliable?) and measuring linkage precision/recall on each release.
- **The future learned scorer** — a gradient-boosted model that will replace the hand-tuned scoring weights. It needs both positive examples (matches) and hard negatives (high-scoring `no_match` pairs) to train.
- **Downstream consumers** — anyone applying copyright reasoning to the published linkage dataset.

---

## References

- **U.S. Copyright Office — Circular 23, "Copyright Office Records"**: <https://www.copyright.gov/circs/circ23.pdf>. Authoritative breakdown of which copyright records exist for which years and how to access them. Confirms that **December 31, 1977 is the last day of registrations under the 1909 Copyright Act** — the upstream-coverage boundary the project's `_CCE_MAX_YEAR = 1977` reflects (the CCE itself ends there; records from 1978 onward live only in the 1976-Act catalog at copyright.gov, in a different format entirely).
- **Internet Archive — copyright records collection**: <https://archive.org/details/copyrightrecords>. OCR / scans of the same CCE volumes NYPL transcribed. Same source data, less structured form; doesn't extend our coverage. Useful as a human-readable cross-reference when investigating a specific record.
- **Cornell University Library — "Copyright Term and the Public Domain in the United States"**: <https://guides.library.cornell.edu/copyright/publicdomain>. Reference matrix for downstream consumers applying copyright reasoning to the linkage dataset; this project does not itself encode it.

---

## Glossary

| term | meaning |
|---|---|
| **MARC** | MAchine-Readable Cataloging — the dominant library bibliographic format. Princeton's `bibdata` exports their catalog as **MARCXML**. A "record" describes one bibliographic item (book, score, recording, …); fields are numbered (245 = title, 008 = fixed-length metadata, etc.). |
| **CCE** | Catalog of Copyright Entries — the U.S. Copyright Office's published register of copyright registrations and renewals, 1891–1977. The matching authority for pre-1978 U.S. copyright status. |
| **NYPL** | The New York Public Library transcribed the CCE volumes into structured XML/TSV. We consume their transcriptions, not the original LoC PDFs. |
| **LMDB** | Lightning Memory-Mapped Database — the on-disk key-value store the matcher uses for its CCE index. Fast random reads across multiple worker processes; built once by `pd-matcher index build`. |
| **Moving wall** | The lower bound on publication years the matcher cares about. Anything ≤ `today.year − 95` is already PD by age — there is no meaningful linkage signal to record — so it's filtered at acquire time, and the wall advances every January 1. As of 2026, the wall is 1931. |
| **Confidence band** | A discrete bucket the matcher assigns to each score: `ge90` (≥ 0.90), `b80_90`, `b70_80`, `b60_70`, `below` (< 0.60). Labels stratify across bands so we don't only label easy high-confidence pairs. |
| **Stratified sample** | A sampling scheme that takes a *fixed* number of items from each `(language, band)` cell, instead of sampling proportionally. Spreads labeling effort across the score range and across all languages. |
| **Pair** | One `(MARC record, CCE registration)` candidate to be labeled. The matcher proposes; the human disposes. |
| **Vault** | `data/label_vault.jsonl` — the durable, git-tracked, append-only record of every human verdict ever rendered. Source of truth for the ground truth. |

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

The labeling subsystem (`pd-groundtruth`) brings in additional optional dependencies (`fastapi`, `uvicorn`, `jinja2`, `requests`). They're already in the dev install; if you want a leaner production install of just the matcher, use `pip install pd-matcher` and add `[acquire]`, `[review]`, or `[all]` extras as needed.

---

## Quick start (matching)

For the full guided walkthrough — install, index build, acquire, labeling, eval, maintenance — see [docs/USER_GUIDE.md](docs/USER_GUIDE.md). The shortest path from a fresh clone to a working match:

```bash
# Install (once per clone)
pdm install
pdm run pre-commit install

# Build the CCE index (one-time, ~37 s for the full corpus)
pdm run pd-matcher index build \
  --reg-dir data/nypl-reg/xml \
  --ren-dir data/nypl-ren/data \
  --out caches/cce.lmdb

# Match a MARC file against the index
pdm run pd-matcher match \
  --marc data/candidate_marc_file.marcxml \
  --index caches/cce.lmdb \
  --out results.csv
```

The first time `pd-matcher match` runs against an index, it builds (and caches) a token-IDF table from the CCE titles. Subsequent runs reuse the cache.

---

## Matcher commands (`pd-matcher`)

### `pd-matcher index build`

Build the LMDB-backed CCE index. Streams the registration XML and renewal TSV files, normalizes each record, precomputes the registration↔renewal join (so workers never join at match time), and writes one mmap'd LMDB environment.

```bash
pd-matcher index build \
  --reg-dir data/nypl-reg/xml \
  --ren-dir data/nypl-ren/data \
  --out caches/cce.lmdb \
  [--force]
```

- `--force` rebuilds even when the existing index is current. Without it, the build short-circuits when the source-file hash, the schema version, **and** a content hash of the parser/model/codec modules (`parsers/nypl_reg.py`, `parsers/nypl_ren.py`, `models.py`, `index/codec.py`, `index/builder.py`) all match the inputs. Any drift in those code files invalidates the cache automatically, so a code change to e.g. `NyplRegRecord` triggers a rebuild without anyone having to bump `schema_version`. The rebuild log line names the mismatch reason (`source_hash_changed`, `schema_version_changed`, `parser_fingerprint_changed`, ...) for debugging.

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
  [--min-score F]
```

- `--workers N` — number of worker processes. Defaults to `cpu_count - 1`. On a 32-core box, that's 31 matching processes plus one for the writer plus the producer and reporter in main.
- `--year-window N` — overrides the matching config's year window (default 2). The matcher only considers CCE records published within `±N` years of the MARC record.
- `--min-score F` — overrides the matching config's minimum calibrated score (default 0.70). Below this threshold a candidate is dropped from the result.

The match command streams: it never holds the whole MARC file in memory. Workers share the LMDB index via the OS page cache, so memory overhead grows with worker count only by a small fixed amount per process.

`Ctrl-C` triggers a graceful shutdown: workers finish their current record, the queue drains, the CSV file is flushed and closed cleanly, and a partial-results path is printed. A second `Ctrl-C` aborts hard.

### `pd-matcher eval`

Run the matcher against the live label vault and produce linkage P/R, AUC, average precision, and a threshold sweep.

```bash
pd-matcher eval \
  --vault data/label_vault.jsonl \
  --pool data/candidates \
  --index caches/cce.lmdb \
  [--report eval.json] \
  [--year-window N]
```

- `--vault PATH` — the JSONL label vault to evaluate against (default `data/label_vault.jsonl`).
- `--pool PATH` — the MARC pool the vault entries reference (default `data/candidates`). Vault entries whose MARC is no longer in the pool are dropped with a logged warning (data drift).
- `--index PATH` — the LMDB CCE index.
- `--report PATH` — write the full `EvalReport` as JSON in addition to the human-readable summary.
- `--year-window N` — override the matching config's year window for this run. Accepted range is 0–100.

Two passes:
- **Per-MARC linkage** — for each unique MARC with a current `match` verdict in the vault, run `match_record` and check whether the matcher's top pick equals the labeled CCE UUID. Produces precision, recall, F1 (gated by `pdm run regression`).
- **Pair-level discrimination** — score every non-`unsure` vault entry directly, collect `(score, label)` pairs, and compute AUC, average precision, and a 21-point threshold sweep across `0.00`–`1.00` step `0.05`. Reported, not currently gated.

The vault is the sole eval corpus as of #25; the prior `combined_ground_truth.csv` workflow has been retired.

### `pd-matcher train-scorer`

Placeholder for the learned scorer (#4). Will eventually train a LightGBM model on the per-Evidence feature vectors and persist it alongside the index. Currently exits 2 with a "not yet implemented" message.

### Global flags

- `--log-level DEBUG|INFO|WARNING|ERROR` (default INFO).
- `--json-logs` — emit structured JSON logs instead of human-readable lines. Useful for piping to log aggregators.
- `--quiet` — suppress everything below WARNING; overrides `--log-level`.

---

## Output format

The CSV is a flat linkage row: MARC metadata on the left, matched CCE metadata on the right, with per-field and combined scores.

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

When no candidate clears the threshold, the `match_*` columns are blank.

---

## How matching works

Matching MARC records to CCE registrations is hard for non-obvious reasons. ISBNs barely existed during the CCE period. LCCNs are present in some MARC records but absent from the CCE side. Titles and authors drift between sources (transcription errors, abbreviation differences, embedded edition info, language conventions). The corpus is multilingual. Years drift by 1–2 between publication, registration, and renewal. A naïve "for every MARC record, score it against every CCE record" pass would be ~2.17M × millions = trillions of comparisons.

The tool solves this in four layers:

**1. Blocking by year.** Every CCE record is indexed into a year bucket. For a MARC record published in 1955, the matcher only retrieves candidates with `reg_year ∈ [1953, 1957]`. The year window is configurable; default ±2.

**2. Per-field scoring.** Each candidate is scored against the MARC record by a set of pure-function scorers — one per signal (title, author, publisher, year, LCCN, ISBN, edition). Each scorer emits a structured `Evidence` object containing its score and named sub-features. Skipped scorers (missing fields) contribute nothing rather than penalizing.

**3. Configurable field pairings.** Title, author, and publisher are transposable across the two sources (the work title stored as a series title, the publisher as the copyright claimant, the author present only in the 245 statement of responsibility). For each of these groups the matcher tries several `(MARC field, CCE field)` pairings and keeps the highest-scoring Evidence; the runners-up are preserved for audit. The pairing set is **configuration**, not code — see [`src/pd_matcher/config/defaults/field_pairings.yaml`](src/pd_matcher/config/defaults/field_pairings.yaml) and the [field-pairings study](docs/studies/field-pairings.md). Code surfaces raw subfields out of each record; a small closed vocabulary in YAML composes and pairs them, validated at load time.

**4. Combination + calibration.** A weighted-mean combiner reduces the Evidence collection to a single raw score. A Platt-scaled logistic regression, trained against the project's ground-truth set, maps the raw score to a calibrated probability. A "75" in the published `combined_score` column means there's roughly a 75% chance the pair is a true match, not "I gave this 75 vibe points."

The matching pipeline is parallelized via Python's `multiprocessing` with the `spawn` start method. Workers share the LMDB index through the OS page cache (memory-mapped reads, zero-copy across processes). The producer streams MARC records from disk, batches them, and feeds workers via a bounded queue (backpressure). A single writer process consumes results and serializes the CSV. A reporter thread aggregates throughput and ETA.

Subfield values from MARC, CCE registrations, and CCE renewals are routed through [ftfy](https://ftfy.readthedocs.io/) at parse time to repair mojibake (``cafÃ©`` → ``café``, ``Â© 2020`` → ``© 2020``), strip stray BOMs, and remove bidirectional formatting marks that would otherwise split tokens. CCE renewals (read as raw bytes) additionally have a Windows-1255 fallback decoder for any Hebrew content that fails strict UTF-8 — currently unused in the supplied corpus but present for future ingests. Per-parser counters (``MarcParseStats``, ``NyplRegParseStats``, ``NyplRenParseStats``) expose how many cells were repaired or routed through each fallback.

For a deep dive into the algorithm, scoring math, and design rationale, see [design.md](docs/design.md). For how candidate **retrieval** (inverted token indexes + year buckets) is separated from **scoring**, and the performance work behind it, see [matching-architecture.md](docs/matching-architecture.md). For the per-branch shipping workflow — gates, regenerated baseline, per-pair diff against `main`, merge — see [phase-workflow.md](docs/phase-workflow.md).

---

## Performance

On a 32-core machine with the full submodule corpus:

- **Index build**: ~37 seconds.
- **Match throughput**: roughly 1–3k MARC records per second per worker depending on field complexity and number of year-bucket candidates per record. With 31 workers, a million-record MARC file finishes in tens of seconds, dominated by the LMDB read and the rapidfuzz scorers.
- **Memory**: each worker process is small (~50 MB Python overhead + ~10 MB scorer state); the LMDB file is mapped once and shared via the kernel page cache.

The `slow` pytest marker excludes one full-corpus integration test from the default test run; invoke it with `pdm run pytest -m slow` to verify end-to-end behavior against the real submodules.

---

## Configuration

The matcher ships with shippable-default YAML files:

- `src/pd_matcher/config/defaults/matching.yaml` — scorer weights, year window, min combined score, scorer selection.
- `src/pd_matcher/config/defaults/field_pairings.yaml` — the `(MARC field, CCE field)` pairings tried for the title, author, and publisher scorer groups, composed from raw subfields via a closed combine vocabulary. Documented inline; see the [field-pairings study](docs/studies/field-pairings.md).

Both load through `src/pd_matcher/config/loader.py` against `msgspec.Struct` schemas in `src/pd_matcher/config/schemas.py`. Schema-violating YAML fails loudly at load time. Future work will let a user-provided YAML override or merge with the defaults; for now, edit the defaults directly if you need to tune.

---

## Ground-truth labeling (`pd-groundtruth`)

The `pd-groundtruth` CLI builds the public-domain **ground-truth corpus**: a set of human-verified `(MARC record, CCE registration)` pairs labeled **match**, **no_match**, or **unsure**. Those labels are what we calibrate and evaluate the matcher against.

The labeling workflow itself — what each verdict means, what to capture in the free-text note, how to handle translations and reprints — is in [`LABELING_GUIDE.md`](LABELING_GUIDE.md).

### Commands at a glance

| command | purpose |
|---|---|
| `pdm run pd-groundtruth acquire …` | Download Princeton's MARC dumps and filter to books in scope (language, year, format). Writes MARCXML shards. |
| `pdm run pd-groundtruth build-queue …` | Run the matcher on the filtered shards, stratified-sample the resulting pairs, and write a `review.db`. Auto-includes every in-pool vault label. |
| `pdm run pd-groundtruth review …` | Serve the local web UI for labeling pairs. Every verdict writes both to `review.db` and to the vault. |
| `pdm run pd-groundtruth seed-vault …` | One-shot migration: dump every current label from a pre-existing `review.db` into the vault. Idempotent. |
| `pdm run pd-groundtruth vault-into-queue …` | Recovery tool: backfill an existing `review.db` with vault entries that aren't already present. Rarely needed after the build-queue carryover fix; see "Recovery" below. |
| `pdm run pd-groundtruth migrate-vault-v3 …` | One-shot migration: fold pre-schema-3 `reasons` / `field_annotations` into the note text and rewrite the vault at the current schema. Idempotent. |
| `pdm run pd-groundtruth dump-vault-marcs …` | Write a MARCXML `<collection>` containing every MARC record referenced by the vault, drawn from `data/candidates/`. The MARC half of the published linkage dataset; pairs with `data/label_vault.jsonl`. Default output is `data/published/vault_marcs.xml` — a path inside an in-tree clone of the separate [`jpstroop/cce-marc-linkage`](https://github.com/jpstroop/cce-marc-linkage) data repo (gitignored from this repo). Read-only against the vault and the pool; safe to run anytime. |

### How it fits together

Building the corpus is a three-stage pipeline. Each stage is one command, and each hands a concrete artifact to the next:

```
Princeton bibdata          CCE index
  MARC dumps              (caches/cce.lmdb,
     │                     built by pd-matcher)
     │                          │
     ▼                          ▼
  ┌────────┐   MARCXML    ┌─────────────┐   review.db   ┌────────┐   labels
  │acquire │ ──shards──▶  │ build-queue │ ──(SQLite)──▶ │ review │ ──in──▶ ground
  └────────┘              └─────────────┘               └────────┘  review.db   truth
```

1. **`acquire`** downloads Princeton's `bibdata` MARC dumps and keeps only the records worth labeling (books, in supported languages, published in the window where copyright status is actually in question — see [What survives acquisition](#reference-what-survives-acquisition)). Survivors are written as MARCXML shards under `<out-dir>/<lang>/`.

2. **`build-queue`** runs those shards through the matcher (the *same* engine the production matcher uses), finds each record's best CCE candidate, and writes a **stratified sample** of the resulting pairs into a `review.db` SQLite file. Stratifying by `(language, confidence band)` spreads the labeling effort across the whole score range instead of piling onto easy high-confidence pairs — which is what makes the labels useful for tuning thresholds.

3. **`review`** serves a local, keyboard-driven web UI over `review.db`. You judge each proposed pair; every verdict is written back into the same database. The accumulating set of verdicts **is** the ground truth.

You need a built CCE index before stage 2. Build it from the repo root with `pdm run pd-matcher index build …`.

### Usage

Every command below writes a log file to `logs/{command}_{utc-timestamp}.log` in addition to streaming the same lines to the console. Override the path with `--log-file PATH` (the parent directory is created if missing). The `logs/` directory is gitignored.

#### 1. `acquire` — download and filter MARC

```bash
pdm run pd-groundtruth acquire --out-dir data/candidates
```

| flag | default | meaning |
|---|---|---|
| `--out-dir` | *(required)* | Root directory; shards are written to `<out-dir>/<lang>/`. |
| `--manifest-url` | `https://bibdata.princeton.edu/dumps/16368.json` | The Princeton **dump manifest**: a JSON document listing every MARC dump file (download URL + md5) that makes up one full bibliographic export. `acquire` reads this list, then streams each dump in turn. Override only to target a different/newer export. |
| `--per-decade-cap` | `20000` | Maximum records to keep per `(language, decade)` bucket, so no single decade dominates a language's slice. |
| `--min-year` | the moving wall (`today.year − 95`) | Lower bound on publication year. Defaults to the wall and is recomputed each run; pass an explicit year only for reproducible/replay runs. |
| `--max-dumps` | *(all)* | Stop after processing this many dumps. Handy for a quick smoke run. |

A normal full run is just `--out-dir data/candidates` — every other flag has a sensible default. The run stops when every `(language, decade)` bucket is full, or the dumps are exhausted, or `--max-dumps` is hit.

Each completed dump logs one progress line — English's per-decade fill plus a running total for the other languages:

```
dump done: scanned=124301 running_total=124301 eng [1930]=4101/20000 [1940]=8800/20000 [1950]=12000/20000 [1960]=20000/20000 [1970]=20000/20000 | fre total=412 ger total=380 spa total=151 ita total=77
```

The first time a `(language, decade)` bucket fills it logs `bucket full: eng[1960] reached quota 20000`, and the run ends with a summary table of every per-language per-decade fill plus the stop reason.

#### 2. `build-queue` — match and stratify into a review queue

```bash
pdm run pd-groundtruth build-queue \
  --pool data/candidates \
  --index caches/cce.lmdb \
  --out data/review.db
```

| flag | default | meaning |
|---|---|---|
| `--pool` | *(required)* | The `acquire` output directory whose `<lang>/*.xml` shards form the candidate pool. |
| `--index` | *(required)* | The LMDB index produced by `pd-matcher index build`. |
| `--out` | *(required)* | Destination `review.db` SQLite file. |
| `--vault` | `data/label_vault.jsonl` | The JSONL **label vault** (see below). Vault verdicts are pre-applied to the new queue. |
| `--rebuild` | off | Delete an existing `--out` database before writing. Destructive; required when the target already contains pairs and you want a clean rebuild. |
| `--append` | off | Append into a non-empty `--out` database instead of erroring. Mutually exclusive with `--rebuild`. |
| `--budget` | *(fills default caps)* | Target total number of pairs; scales the per-stratum caps proportionally. |
| `--sample-per-lang` | `1500` | Reservoir size drawn from each language directory before matching. |
| `--workers` | `8` | Number of spawn-pool worker processes. |
| `--seed` | `42` | Seed for the reservoir samplers (reproducible queues). |
| `-v` / `-vv` | off | `-v` adds per-worker throughput heartbeats (records/sec + ETA); `-vv` logs every match hit. |

If `--out` already contains `review_pair` rows and neither `--rebuild` nor `--append` is set, `build-queue` exits with code 2 and an actionable message rather than silently merging old and new pairs.

On completion it prints `records_sampled`, `records_matched`, `pairs_written`, and the per-stratum counts.

#### 3. `review` — label the pairs in the web UI

```bash
pdm run pd-groundtruth review --db data/review.db
```

| flag | default | meaning |
|---|---|---|
| `--db` | *(required)* | The `review.db` produced by `build-queue`. |
| `--vault` | `data/label_vault.jsonl` | The JSONL **label vault** (see below). Every accepted verdict is appended here in addition to `review.db`. |
| `--host` | `127.0.0.1` | Interface to bind the local server. |
| `--port` | `8000` | Port for the local server. |

Open <http://127.0.0.1:8000>. Ctrl-C stops the server; labels persist in the database, so you can stop and resume any time.

**Each card** shows the MARC record (left) against the proposed CCE candidate (right), the per-field evidence bars, the overall score and confidence band, and the **renewal flag** — a registration that was *not* renewed is a useful downstream signal. When a registration was renewed, a **renewal-details** sub-block also appears with the renewal date, the claimants as transcribed on the renewal (with a warning marker when they differ from the registration's claimants), any new-matter the renewal claimed, and the renewal's id / oreg as compact metadata. The CCE panel shows author place, claimant flag, edition, publication places, physical description, new-matter-claimed, copies, notes, copyright date, affidavit date, notice date, the LCCN (linked to lccn.loc.gov for cross-referencing), and any previous registration numbers — every field the parser was able to extract from the CCE source.

**Label with the keyboard.** The UI auto-advances to the next unlabeled pair, and every keypress writes to `review.db`:

| key | verdict |
|---|---|
| `y` | match |
| `n` | no_match |
| `u` | unsure |
| `s` or space | skip (advance without labeling) |

On-screen buttons do the same. To step back and fix a verdict, press `b` (or `←`); it returns to the pair you most recently labeled and chains further back from there.

Skips are **session-local**: each skipped `pair_id` is appended to the URL as a `?skip=<id>` parameter so the next request asks the database for the next unlabeled pair that is not in that list. The state lives in the URL only — close the tab and reopen, and skipped pairs are back in the queue. Labeling clears the skip list (the POST-redirect to `/` carries the language/band filter but drops `skip`), because committing a verdict ends one attention sweep.

**Capture observations in the note field** (optional, never blocks the fast path). A free-text textarea sits directly below the card. Use it to record anything worth flagging about the pair: what surprised you, what made the call ambiguous, where the scorer seemed wrong. Notes are not constrained by a controlled vocabulary; later analysis reads across all notes to surface patterns. Leave it blank when the verdict is obvious.

**Focus a session** with URL filters — useful for the English-first curriculum (label the easier languages before the harder ones):

- `…/?language=eng` (or `fre` / `ger` / `spa` / `ita`)
- add `&band=ge90` (or `b80_90`, `b70_80`, `b60_70`, `below`) to drill into one confidence band

**Track progress** at `…/stats`: labeled vs. remaining and the match / no_match / unsure tally, per language. Revisit or re-label any specific pair at `…/pair/{id}`.

**Spot-check the training set** at `…/labels`: a flat table of every labeled pair (most recent first), with the verdict, the note (truncated; full text on hover), language, and relative time. Filters in the header narrow by `verdict`, `language`, or a free-text substring (`?q=…`); 100 rows per page. Click any `pair_id` to jump to `/pair/{id}` and re-label. This is the tool for catching systematic mistakes — for example, scanning all `match` verdicts to see if any landed on records that shouldn't have been labeled.

**For the decision rules** (when to call something `match` vs `no_match` vs `unsure`, how translations and reprints work, what kinds of observations the note field should capture), read [`LABELING_GUIDE.md`](LABELING_GUIDE.md). The guide is short and opinionated; consult it the first time you sit down to label and any time a new edge case appears.

### The label vault

`review.db` is a **transient working queue** — it is rebuilt every time `acquire` or `build-queue` runs (for example, when a new filter lands in `acquire`). To stop those rebuilds from destroying the human labels already adjudicated, every verdict is also persisted to a durable **label vault**: `data/label_vault.jsonl`. The vault is the canonical ground-truth dataset and is committed to git.

#### Roles at a glance

| | `data/review.db` | `data/label_vault.jsonl` |
|---|---|---|
| **Purpose** | Working queue for the review web UI | Canonical, durable record of human verdicts |
| **Format** | SQLite (binary) | JSONL (one verdict per line, append-only) |
| **Lifetime** | Transient — rebuilt by `build-queue` | Permanent — committed to git |
| **Holds** | Candidate pairs + labels for *this* run | Every verdict ever rendered, across runs |
| **Source of truth?** | No (derived) | **Yes** |
| **Survives a pipeline rerun?** | No (overwritten) | Yes (append-only) |
| **What to publish** | Never | The matches subset: `jq -c 'select(.verdict=="match")' data/label_vault.jsonl` |

When `build-queue` rebuilds `review.db`, it consults the vault and pre-applies every known verdict, so the working queue is `(new candidate pool) − (already labeled)`. The vault grows monotonically; `review.db` is disposable.

- **Format.** JSONL, append-only, one verdict event per line, schema-versioned (each row carries `"schema": 3`). Every line records the `(marc_control_id, nypl_uuid)` pair, the verdict, the optional free-text note, the ISO-8601 `labeled_at` timestamp, the labeler, and the MARC identifiers (`lccn`, `oclc`, `isbns`) captured at label time.
- **Multi-label semantics.** Re-labels do not overwrite — they append a new line. The "current" verdict for a pair is the last entry by file order; earlier entries are kept as the audit trail.
- **`review` writes here on every POST.** After the DB write succeeds, the same verdict is appended to the vault using the DB-stamped timestamp so the two stay in lockstep. A rare vault-write failure is logged but never fails the HTTP request.
- **`build-queue` always carries the vault forward.** At the start of the run, every current vault entry is resolved against the candidate pool and the LMDB index, scored with the matcher's per-pair routine, and queued for unconditional insertion into the new `review.db` with the original verdict pre-applied (preserving `labeled_at`). Vault MARCs are then *excluded* from the per-language reservoir so the matcher doesn't propose a competing pair for the same record. The result: a rebuilt queue carries every persistable vault verdict regardless of `--sample-per-lang`. The end-of-run summary reads `vault pre-applied: P pairs (of M resolved); Q non-vault pairs queued`. A vault entry whose MARC is no longer in the pool, or whose CCE is no longer in the index, is logged with a WARNING and skipped for this build — the vault file itself is never modified, so a future build can pick it up if the underlying data returns. Use the back-arrow to revisit and re-label if needed.

#### Data model

**`review.db`** (SQLite) — two tables:

```sql
review_pair (id PK, language, decade, score, band, source,
             marc_control_id, marc_json, marc_title, marc_author,
             marc_publisher, marc_year,
             nypl_uuid, cce_title, cce_author, cce_publishers,
             cce_claimants, cce_reg_year, cce_was_renewed, cce_regnum,
             cce_edition, cce_publication_places, cce_author_place,
             cce_author_is_claimant, cce_copies, cce_aff_date, cce_desc,
             cce_notes, cce_new_matter_claimed, cce_copy_date,
             cce_notice_date, cce_lccn, cce_prev_regnums,
             cce_renewal_id, cce_renewal_oreg, cce_renewal_rdat,
             cce_renewal_author, cce_renewal_title,
             cce_renewal_claimants, cce_renewal_new_matter,
             evidence_json, evidence_sources_json, created_at)
label       (id PK, pair_id FK→review_pair, verdict, note,
             labeled_at)
```

Notes:

- `review_pair.marc_json` is the lossless serialized `MarcRecord` from the matcher; the denormalized `marc_*` columns are convenience copies for cheap list-rendering.
- `evidence_json` is the per-scorer `{name: normalized_score}` map the matcher produced for that pair (drives the evidence bars in the card).
- `label` is **append-only**. Re-labeling a pair inserts a new row; the "current" verdict for a pair is the one with the largest `label.id` for that `pair_id` (or equivalently, the latest `labeled_at` with a tie-break on insertion order).

Legacy `label_reason` and `label_field_annotation` tables created before schema 3 may still exist in older `review.db` files but the new code does not query them. Rebuild with `build-queue --rebuild` to get a clean schema; the migration of historical signal lives on the vault side (see `migrate-vault-v3`).

**`data/label_vault.jsonl`** — one JSON object per line, append-only:

```json
{
  "schema": 3,
  "marc_control_id": "9912345678906421",
  "nypl_uuid": "129B8D87-6CB2-1014-A20E-B9D6251C946A",
  "verdict": "match",
  "note": "title matches verbatim; publisher differs but both are Macmillan imprints",
  "labeled_at": "2026-05-23T12:59:45.522659+00:00",
  "labeler": "jpstroop",
  "marc_identifiers": {
    "lccn": "58059853",
    "oclc": null,
    "isbns": []
  }
}
```

Field semantics:

- `schema` — integer; bumped on breaking shape changes. The current schema is 3. Pre-schema-3 lines carried `reasons` and `field_annotations` arrays; running `pdm run pd-groundtruth migrate-vault-v3` folds those into the note text (prefixed as `[reasons: ...]` and `[annotations: field:judgment, ...]`) and rewrites the file at schema 3, archiving the original to `data/label_vault.jsonl.pre-v3`.
- `(marc_control_id, nypl_uuid)` — the natural key. Multiple entries with the same key represent a re-label history; the **last** line wins as the current verdict for that pair.
- `note` — optional free text; the only structured signal carried alongside the verdict. See [`LABELING_GUIDE.md`](LABELING_GUIDE.md) for what's worth capturing.
- `labeler` — string identifier of who labeled (today, always `"jpstroop"`; reserved for future multi-reviewer setups).
- `marc_identifiers` — durable IDs captured at label time so the published matches dataset can cross-walk to LCCN / OCLC / ISBN later.

The vault is **the** source of truth. `review.db` is a queryable, derivable working copy that the labeling app needs for fast pair-by-pair access; the vault is what survives every rebuild.

#### One-shot migration: `seed-vault`

A one-time-only command that exports every *current* label from a pre-existing `review.db` into the vault. Run it once before the next `build-queue` rebuild:

```bash
pdm run pd-groundtruth seed-vault --db data/review.db --vault data/label_vault.jsonl
```

`seed-vault` is idempotent: it skips entries already present with the same `(marc_control_id, nypl_uuid, labeled_at)` triple, so re-running it after a fresh round of labeling only adds the new events.

#### Recovery: `vault-into-queue`

After the build-queue carryover fix (jpstroop/pd-matcher#33), this command is rarely needed in normal operation — it remains available as a recovery tool. Use it when an existing `review.db` is missing vault entries it should contain, for example because the queue was built with vault carryover disabled, or because the vault file was modified out of band after the build.

```bash
pdm run pd-groundtruth vault-into-queue \
  --db data/review.db \
  --vault data/label_vault.jsonl \
  --pool data/candidates \
  --index caches/cce.lmdb
```

| flag | meaning |
|---|---|
| `--db` | Existing review database to backfill in place. |
| `--vault` | JSONL label vault whose entries seed the missing set. |
| `--pool` | The `acquire` output directory; needed to materialize the MARC record for each missing entry. |
| `--index` | The LMDB index produced by `pd-matcher index build`; needed to materialize the CCE registration for each missing entry. |

For each missing entry, the command looks the MARC up in `--pool`, looks the CCE registration up in `--index`, scores the **specific** pair with the matcher's per-pair scoring routine so the row carries real `(score, band, evidence)`, and inserts both the `review_pair` row and the pre-existing vault verdict (preserving the original `labeled_at`). Vault entries whose MARC is no longer in the pool or whose CCE is no longer in the index are logged with a WARNING and skipped; the vault file is never modified.

The final summary reads `backfilled N vault pairs; M MARC records not found in pool; K CCE records not found in index; P already present (skipped)`.

### Common workflows

Recipes for situations that come up in practice. Each is a short, self-contained sequence; nothing here is required reading.

#### First-time setup

Build a fresh CCE index, acquire MARC, build the queue, and start labeling.

```bash
# from the repo root: build the CCE index once
pdm run pd-matcher index build --out caches/cce.lmdb \
    --reg-dir data/nypl-reg/xml --ren-dir data/nypl-ren/data

pdm install
pdm run pd-groundtruth acquire     --out-dir data/candidates
pdm run pd-groundtruth build-queue --pool data/candidates \
                                   --index caches/cce.lmdb \
                                   --out  data/review.db
pdm run pd-groundtruth review      --db   data/review.db
```

Open <http://127.0.0.1:8000> and start labeling.

#### Resume labeling tomorrow

Nothing special; the DB and vault hold all state.

```bash
pdm run pd-groundtruth review --db data/review.db
```

#### Rebuild the candidate set after an upstream change

When the e-book filter changes, Princeton publishes a new bibdata snapshot, the moving wall advances, or the matcher's scoring changes — pull fresh MARC and rebuild the queue. The vault auto-carries-over your existing labels.

```bash
rm -rf data/candidates
pdm run pd-groundtruth acquire     --out-dir data/candidates
pdm run pd-groundtruth build-queue --pool   data/candidates \
                                   --index  caches/cce.lmdb \
                                   --out    data/review.db \
                                   --rebuild
pdm run pd-groundtruth review      --db     data/review.db
```

`--rebuild` is required when `data/review.db` already contains pairs; `build-queue` refuses to silently append (which is how 28 e-book records contaminated a previous queue). Use `--append` if you actually want the old behavior.

#### Spot-check the training set for a systematic mistake

Open <http://127.0.0.1:8000/labels>. Filter the table by verdict (`?verdict=match`) or language (`?language=eng`). Use the `?q=` substring search to find by title or control ID. Click any `pair_id` to jump into `/pair/{id}` and re-label.

#### Purge a bad subset of labels from the vault

If you discover you've labeled records that violate scope (e.g., an entire class of records that should have been filtered out at `acquire` time and slipped through), the vault is the source of truth and needs cleaning. Pattern:

1. Identify the offending vault keys `(marc_control_id, nypl_uuid)`. Often you can derive them from `review.db` via the `marc_json.extent` or another denormalized column.
2. Archive the vault before mutating it.
3. Rewrite the vault, keeping only entries whose key is **not** in the offender set.
4. Rebuild the queue (`build-queue --rebuild`); vault carryover applies only the surviving labels.

There is no built-in `vault prune` subcommand today; the operation is intentionally manual because purges are rare and you should look at what you're removing. A worked example lives in the commit message of the "vault: purge 28 e-book entries; rebuild from clean 129" commit.

#### Revisit a single previously labeled pair

Navigate to `/pair/{id}` (or click the pair ID from the `/labels` table). The card looks the same as a fresh one; labeling it again appends a new row to `label` (and a new line to the vault). The "current" verdict is the latest, but the history is preserved.

#### Reset everything and start fresh (rare)

Deletes the labeled corpus. Don't do this unless you're sure.

```bash
rm data/review.db data/label_vault.jsonl
```

Followed by a fresh acquire + build-queue + review.

### Reference: what survives acquisition

This section explains the *why* behind `acquire`'s filtering and sampling. You do not need it to run the pipeline.

#### Filter criteria

A record survives only if **all** of the following hold (decided directly off the raw leader, 008, and 245):

1. **Monograph book** — leader position 6 is `a` and position 7 is `m`.
2. **Not an electronic resource** — excluded if MARC 007 byte 0 is `c`, MARC 338 $b is `cr`, MARC 245 $h contains "electronic resource", or MARC 300 $a contains "online resource". These are digital reprints of older works (the MARC describes the digital reissue, not the original artifact registered at the Copyright Office).
3. **Supported language** — 008 positions 35:38 are one of `eng`, `fre`, `ger`, `spa`, `ita` (the languages the CCE index covers).
4. **Publication year in window** — 008 positions 7:11 parse as a 4-digit integer in the inclusive range `[min_year, 1977]`. Partial/unknown values (`uuuu`, `||||`, blanks) are rejected. The lower bound is the **moving wall** (below); `1977` is the last CCE renewal year of interest.
5. **Not a government publication** — 008 position 28 must be blank (`" "`) or `"|"`; any coded value (`a c f i l m o s u z` …) is dropped. Government works are public domain by statute and were never registered in the CCE, so they are pure noise — an early live run found ~95% of survivors were government publications, drowning out the records we need.
6. **Has a title** — a 245 data field with a non-empty subfield `a`.

Records are matched by element *local name*, so both the MARC21 slim namespace and the no-namespace serialization work.

#### The moving wall

Works published at or before `today.year − 95` are already public domain by age (= 1931 as of 2026) and carry no copyright-status signal, so keeping them would only dilute the corpus. Because the bound is computed per run, it advances every January 1 without a code change. `--min-year` overrides it for replay runs.

#### Disk streaming

Acquisition never materializes a full dump in memory or on disk beyond a single compressed archive:

- each dump is streamed to a temporary `.tar.gz`, its md5 verified against the manifest, then **deleted** before the next dump starts;
- the archive is opened in streaming mode (`tarfile.open(mode="r|gz")`) and its single member is fed straight to `lxml.etree.iterparse`;
- every record is `clear()`-ed after inspection, bounding memory to roughly one record at a time.

Disk peak is therefore ~one compressed dump file.

#### Per-(language, decade) quotas

Sampling is constrained per `(language, decade)` rather than by a flat per-language cap. A record's decade bucket is `(year // 10) * 10`, giving buckets `1930, 1940, 1950, 1960, 1970` (the `1930` bucket holds only `1931–1939` under the wall, and `1970` holds `1970–1977`). The bucket set is derived from `min_year..1977`, so it stays correct as the wall moves.

`--per-decade-cap` applies to every `(language, decade)` pair: a record is kept only while its own bucket is below the quota. The non-English buckets essentially never fill, so in practice the run scans every dump — intentional, to gather every available non-English book and as many English-per-decade as exist.

#### Output layout

Shards are written under `out_dir/<lang>/`, each a valid `<collection>` capped at 5000 records:

```
data/candidates/
  eng/candidates_00001.xml
  eng/candidates_00002.xml
  fre/candidates_00001.xml
  ger/candidates_00001.xml
  spa/candidates_00001.xml
  ita/candidates_00001.xml
```

A language subdirectory is created only when at least one record lands in it. The decade is a **sampling constraint only** — there are no decade subdirectories; records of all decades for a language are interleaved across that language's shards.

---

## Development

```bash
pdm run gates              # ruff format + ruff check + mypy + pytest with 100% line+branch coverage
pdm run fmt                # ruff format
pdm run lint               # ruff check
pdm run lint-fix           # ruff check --fix (autofix where safe)
pdm run typecheck          # mypy
pdm run test               # pytest (excludes slow, regression, and webui tests by default)
pdm run webui              # FastAPI review-UI route+template smoke tests (deselected from default gate)
pdm run pytest -m slow     # run only the slow tests
pdm run regression         # index-dependent eval gate vs tests/regression/baseline.json (fails on >2pp precision/recall drop; skips if the index is absent)
pdm run regression-baseline # refresh tests/regression/baseline.json after an intentional pipeline change
```

The pre-commit hook runs `ruff format`, `ruff check --fix`, end-of-file fixer, trailing-whitespace fixer, and merge-conflict detection on every `git commit`. The heavier gates (mypy, pytest) are deliberately *not* in the pre-commit chain — they run via `pdm run gates` and (eventually) CI.

Project standards:

- Strict mypy with `disallow_any_explicit = true`. No `Any`. No `# type: ignore`.
- One import per line (`from os import getpid`, not `import os` and not `from os import getpid, unlink`).
- One class per file (private helpers may colocate).
- msgspec `Struct` for typed records throughout.
- 100% line + branch test coverage enforced (matcher + groundtruth core; the FastAPI app/server modules are excluded and covered by the `webui` route-level smoke tests instead).

See [design.md](docs/design.md) for the technology decisions and what motivates them, and the [glossary](docs/glossary.md) for definitions of the statistics, matching, and tooling terms used throughout. For the human labeler's decision rules, see [LABELING_GUIDE.md](LABELING_GUIDE.md).

---

## License

The `pd-matcher` source code is licensed under the **GNU Affero General Public
License v3.0 or later** (AGPL-3.0-or-later); see [LICENSE](LICENSE). In short:
you may use, modify, and redistribute it, including commercially, but any
distributed or network-deployed modifications must be released under the same
license — closed-source forks and proprietary hosted services are not
permitted.

This license covers the code only, **not the bundled data**:

- The CCE registration and renewal datasets are pulled in as NYPL-transcribed
  git submodules under `data/nypl-reg/` and `data/nypl-ren/`; NYPL's
  transcriptions carry their own licenses.
- The underlying Catalog of Copyright Entries is a work of the U.S. Copyright
  Office (Library of Congress) and is in the public domain in the United States.
- Any MARC catalog you match against, and the ground-truth pairings, are your
  own data under your own terms.
