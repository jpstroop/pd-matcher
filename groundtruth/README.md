# pd-groundtruth

Builds the public-domain **ground-truth corpus** for [`pd-matcher`](..): a set of
human-verified `(MARC record, CCE registration)` pairs labeled **match**,
**no_match**, or **unsure**. Those labels are what we calibrate and evaluate the
matcher against.

It is a standalone PDM subproject because it carries heavier dependencies
(`requests`, `fastapi`, `uvicorn`) and a relaxed coverage bar, so its
configuration never touches the core's strict `pyproject.toml`.

## How it fits together

Building the corpus is a three-stage pipeline. Each stage is one command, and
each hands a concrete artifact to the next:

```
Princeton bibdata          CCE index
  MARC dumps              (caches/nypl.lmdb,
     │                     built by pd-matcher)
     │                          │
     ▼                          ▼
  ┌────────┐   MARCXML    ┌─────────────┐   review.db   ┌────────┐   labels
  │acquire │ ──shards──▶  │ build-queue │ ──(SQLite)──▶ │ review │ ──in──▶ ground
  └────────┘              └─────────────┘               └────────┘  review.db   truth
```

1. **`acquire`** downloads Princeton's `bibdata` MARC dumps and keeps only the
   records worth labeling (books, in supported languages, published in the window
   where copyright status is actually in question — see
   [What survives acquisition](#reference-what-survives-acquisition)). Survivors
   are written as MARCXML shards under `<out-dir>/<lang>/`.

2. **`build-queue`** runs those shards through the matcher (the *same* engine the
   production matcher uses), finds each record's best CCE candidate, and writes a
   **stratified sample** of the resulting pairs into a `review.db` SQLite file.
   Stratifying by `(language, confidence band)` spreads the labeling effort across
   the whole score range instead of piling onto easy high-confidence pairs — which
   is what makes the labels useful for tuning thresholds.

3. **`review`** serves a local, keyboard-driven web UI over `review.db`. You judge
   each proposed pair; every verdict is written back into the same database. The
   accumulating set of verdicts **is** the ground truth.

You need a built CCE index before stage 2. That is a `pd-matcher` artifact, not a
groundtruth one — build it from the repo root with
`pdm run pd-matcher index build …`.

## Setup

```bash
cd groundtruth
pdm install
```

All commands below are run from this `groundtruth/` directory.

## Usage

Every command below writes a log file to `logs/{command}_{utc-timestamp}.log` in
addition to streaming the same lines to the console. Override the path with
`--log-file PATH` (the parent directory is created if missing). The `logs/`
directory is gitignored.

### 1. `acquire` — download and filter MARC

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

A normal full run is just `--out-dir data/candidates` — every other flag has a
sensible default. The run stops when every `(language, decade)` bucket is full, or
the dumps are exhausted, or `--max-dumps` is hit.

Each completed dump logs one progress line — English's per-decade fill plus a
running total for the other languages:

```
dump done: scanned=124301 running_total=124301 eng [1930]=4101/20000 [1940]=8800/20000 [1950]=12000/20000 [1960]=20000/20000 [1970]=20000/20000 | fre total=412 ger total=380 spa total=151 ita total=77
```

The first time a `(language, decade)` bucket fills it logs `bucket full:
eng[1960] reached quota 20000`, and the run ends with a summary table of every
per-language per-decade fill plus the stop reason.

### 2. `build-queue` — match and stratify into a review queue

```bash
pdm run pd-groundtruth build-queue \
  --pool data/candidates \
  --index ../caches/nypl.lmdb \
  --out data/review.db
```

| flag | default | meaning |
|---|---|---|
| `--pool` | *(required)* | The `acquire` output directory whose `<lang>/*.xml` shards form the candidate pool. |
| `--index` | *(required)* | The LMDB index produced by `pd-matcher index build`. |
| `--out` | *(required)* | Destination `review.db` SQLite file. |
| `--rebuild` | off | Delete an existing `--out` database before writing. Destructive; required when the target already contains pairs and you want a clean rebuild. |
| `--append` | off | Append into a non-empty `--out` database instead of erroring. Mutually exclusive with `--rebuild`. |
| `--budget` | *(fills default caps)* | Target total number of pairs; scales the per-stratum caps proportionally. |
| `--sample-per-lang` | `1500` | Reservoir size drawn from each language directory before matching. |
| `--workers` | `8` | Number of spawn-pool worker processes. |
| `--seed` | `42` | Seed for the reservoir samplers (reproducible queues). |
| `-v` / `-vv` | off | `-v` adds per-worker throughput heartbeats (records/sec + ETA); `-vv` logs every match hit. |

If `--out` already contains `review_pair` rows and neither `--rebuild` nor
`--append` is set, `build-queue` exits with code 2 and an actionable message
rather than silently merging old and new pairs.

On completion it prints `records_sampled`, `records_matched`, `pairs_written`,
and the per-stratum counts.

### 3. `review` — label the pairs in the web UI

```bash
pdm run pd-groundtruth review --db data/review.db
```

| flag | default | meaning |
|---|---|---|
| `--db` | *(required)* | The `review.db` produced by `build-queue`. |
| `--vault` | `label_vault.jsonl` | The JSONL **label vault** (see below). Every accepted verdict is appended here in addition to `review.db`. |
| `--host` | `127.0.0.1` | Interface to bind the local server. |
| `--port` | `8000` | Port for the local server. |

Open <http://127.0.0.1:8000>. Ctrl-C stops the server; labels persist in the
database, so you can stop and resume any time.

**Each card** shows the MARC record (left) against the proposed CCE candidate
(right), the per-field evidence bars, the overall score and confidence band, and
the **renewal flag** — the public-domain tell (a registration that was *not*
renewed is the signal we care about).

**Label with the keyboard.** The UI auto-advances to the next unlabeled pair, and
every keypress writes to `review.db`:

| key | verdict |
|---|---|
| `y` | match |
| `n` | no_match |
| `u` | unsure |
| `s` or space | skip (advance without labeling) |

On-screen buttons do the same. To step back and fix a verdict, press `b` (or
`←`); it returns to the pair you most recently labeled and chains further back
from there.

Skips are **session-local**: each skipped `pair_id` is appended to the URL as a
`?skip=<id>` parameter so the next request asks the database for the next
unlabeled pair that is not in that list. The state lives in the URL only — close
the tab and reopen, and skipped pairs are back in the queue. Labeling clears the
skip list (the POST-redirect to `/` carries the language/band filter but drops
`skip`), because committing a verdict ends one attention sweep.

**Record *why* a no-match / unsure** (optional, never blocks the fast path). Each
`no_match` / `unsure` reason is a chip you can toggle on or off; toggle any
number of them (e.g. both "Different work / title collision" and "Garbled
transcription") and then press the verdict key/button to record the verdict
together with every selected code. Codes that don't belong to the chosen verdict
are ignored server-side, so failure modes stay aggregatable. A free-text note
field rides along for anything the codes don't cover. The reason tally shows up
on `…/stats`, counting each code of a pair's current label.

**Focus a session** with URL filters — useful for the English-first curriculum
(label the easier languages before the harder ones):

- `…/?language=eng` (or `fre` / `ger` / `spa` / `ita`)
- add `&band=ge90` (or `b80_90`, `b70_80`, `below`) to drill into one confidence
  band

**Track progress** at `…/stats`: labeled vs. remaining and the match / no_match /
unsure tally, per language. Revisit or re-label any specific pair at `…/pair/{id}`.

## The label vault

`review.db` is a **transient working queue** — it is rebuilt every time
`acquire` or `build-queue` runs (for example, when a new filter lands in
`acquire`). To stop those rebuilds from destroying the human labels already
adjudicated, every verdict is also persisted to a durable **label vault**:
`groundtruth/label_vault.jsonl`. The vault is the canonical ground-truth
dataset and is committed to git.

### Roles at a glance

| | `data/review.db` | `label_vault.jsonl` |
|---|---|---|
| **Purpose** | Working queue for the Tinder app | Canonical, durable record of human verdicts |
| **Format** | SQLite (binary) | JSONL (one verdict per line, append-only) |
| **Lifetime** | Transient — rebuilt by `build-queue` | Permanent — committed to git |
| **Holds** | Candidate pairs + labels for *this* run | Every verdict ever rendered, across runs |
| **Source of truth?** | No (derived) | **Yes** |
| **Survives a pipeline rerun?** | No (overwritten) | Yes (append-only) |
| **What to publish** | Never | The matches subset: `jq -c 'select(.verdict=="match")' label_vault.jsonl` |

When `build-queue` rebuilds `review.db`, it consults the vault and pre-applies
every known verdict, so the working queue is `(new candidate pool) − (already
labeled)`. The vault grows monotonically; `review.db` is disposable.

- **Format.** JSONL, append-only, one verdict event per line, schema-versioned
  (each row carries `"schema": 1`). Every line records the
  `(marc_control_id, nypl_uuid)` pair, the verdict, any reason codes, the
  optional note, the ISO-8601 `labeled_at` timestamp, the labeler, and the
  MARC identifiers (`lccn`, `oclc`, `isbns`) captured at label time.
- **Multi-label semantics.** Re-labels do not overwrite — they append a new
  line. The "current" verdict for a pair is the last entry by file order;
  earlier entries are kept as the audit trail.
- **`review` writes here on every POST.** After the DB write succeeds, the
  same verdict is appended to the vault using the DB-stamped timestamp so the
  two stay in lockstep. A rare vault-write failure is logged but never fails
  the HTTP request.
- **`build-queue` reads here at the start.** Pairs that already have a vault
  verdict are inserted into the new `review.db` *with* their verdict
  pre-applied, so `next_unlabeled` skips them and `/stats` immediately
  reflects the existing labels. A summary line at the end reports
  `vault contributed N pre-applied labels; M new pairs queued`. Use the
  back-arrow to revisit and re-label if needed.

### One-shot migration: `seed-vault`

A one-time-only command that exports every *current* label from a
pre-existing `review.db` into the vault. Run it once before the next
`build-queue` rebuild:

```bash
pdm run pd-groundtruth seed-vault --db data/review.db --vault label_vault.jsonl
```

`seed-vault` is idempotent: it skips entries already present with the same
`(marc_control_id, nypl_uuid, labeled_at)` triple, so re-running it after a
fresh round of labeling only adds the new events.

### Recovery: `vault-into-queue`

A tactical bridge that backfills vault entries whose `(marc_control_id,
nypl_uuid)` pair is **not** present in an existing `review.db`. This happens
because `build-queue`'s stratified sampling can miss MARC records that were
labeled in a previous round, leaving those verdicts invisible in the review
UI even though they are safely persisted in the vault.

```bash
pdm run pd-groundtruth vault-into-queue \
  --db data/review.db \
  --vault label_vault.jsonl \
  --pool data/candidates \
  --index ../caches/nypl.lmdb
```

| flag | meaning |
|---|---|
| `--db` | Existing review database to backfill in place. |
| `--vault` | JSONL label vault whose entries seed the missing set. |
| `--pool` | The `acquire` output directory; needed to materialize the MARC record for each missing entry. |
| `--index` | The LMDB index produced by `pd-matcher index build`; needed to materialize the CCE registration for each missing entry. |

For each missing entry, the command looks the MARC up in `--pool`, looks the
CCE registration up in `--index`, scores the **specific** pair with the
matcher's per-pair scoring routine so the row carries real `(score, band,
evidence)`, and inserts both the `review_pair` row and the pre-existing vault
verdict (preserving the original `labeled_at`). Vault entries whose MARC is
no longer in the pool or whose CCE is no longer in the index are logged with
a WARNING and skipped; the vault file is never modified.

The final summary reads `backfilled N vault pairs; M MARC records not found
in pool; K CCE records not found in index; P already present (skipped)`.

This is a **tactical unblock** until `build-queue` always includes vault MARCs
in its sample (see `jpstroop/pd-matcher#33`); once that lands, this command
becomes unnecessary.

## Reference: what survives acquisition

This section explains the *why* behind `acquire`'s filtering and sampling. You do
not need it to run the pipeline.

### Filter criteria

A record survives only if **all** of the following hold (decided directly off the
raw leader, 008, and 245):

1. **Monograph book** — leader position 6 is `a` and position 7 is `m`.
2. **Not an electronic resource** — excluded if MARC 007 byte 0 is `c`, MARC
   338 $b is `cr`, MARC 245 $h contains "electronic resource", or MARC 300 $a
   contains "online resource". These are digital reprints of older works (the
   MARC describes the digital reissue, not the original artifact registered
   at the Copyright Office).
3. **Supported language** — 008 positions 35:38 are one of `eng`, `fre`, `ger`,
   `spa`, `ita` (the languages the CCE index covers).
4. **Publication year in window** — 008 positions 7:11 parse as a 4-digit integer
   in the inclusive range `[min_year, 1977]`. Partial/unknown values (`uuuu`,
   `||||`, blanks) are rejected. The lower bound is the **moving wall** (below);
   `1977` is the last CCE renewal year of interest.
5. **Not a government publication** — 008 position 28 must be blank (`" "`) or
   `"|"`; any coded value (`a c f i l m o s u z` …) is dropped. Government works
   are public domain by statute and were never registered in the CCE, so they are
   pure noise — an early live run found ~95% of survivors were government
   publications, drowning out the records we need.
6. **Has a title** — a 245 data field with a non-empty subfield `a`.

Records are matched by element *local name*, so both the MARC21 slim namespace
and the no-namespace serialization work.

### The moving wall

Works published at or before `today.year − 95` are already public domain by age
(= 1931 as of 2026) and carry no copyright-status signal, so keeping them would
only dilute the corpus. Because the bound is computed per run, it advances every
January 1 without a code change. `--min-year` overrides it for replay runs.

### Disk streaming

Acquisition never materializes a full dump in memory or on disk beyond a single
compressed archive:

- each dump is streamed to a temporary `.tar.gz`, its md5 verified against the
  manifest, then **deleted** before the next dump starts;
- the archive is opened in streaming mode (`tarfile.open(mode="r|gz")`) and its
  single member is fed straight to `lxml.etree.iterparse`;
- every record is `clear()`-ed after inspection, bounding memory to roughly one
  record at a time.

Disk peak is therefore ~one compressed dump file.

### Per-(language, decade) quotas

Sampling is constrained per `(language, decade)` rather than by a flat
per-language cap. A record's decade bucket is `(year // 10) * 10`, giving buckets
`1930, 1940, 1950, 1960, 1970` (the `1930` bucket holds only `1931–1939` under the
wall, and `1970` holds `1970–1977`). The bucket set is derived from
`min_year..1977`, so it stays correct as the wall moves.

`--per-decade-cap` applies to every `(language, decade)` pair: a record is kept
only while its own bucket is below the quota. The non-English buckets essentially
never fill, so in practice the run scans every dump — intentional, to gather every
available non-English book and as many English-per-decade as exist.

### Output layout

Shards are written under `out_dir/<lang>/`, each a valid `<collection>` capped at
5000 records:

```
data/candidates/
  eng/candidates_00001.xml
  eng/candidates_00002.xml
  fre/candidates_00001.xml
  ger/candidates_00001.xml
  spa/candidates_00001.xml
  ita/candidates_00001.xml
```

A language subdirectory is created only when at least one record lands in it. The
decade is a **sampling constraint only** — there are no decade subdirectories;
records of all decades for a language are interleaved across that language's
shards.

## Development

```bash
cd groundtruth
pdm run gates   # ruff format + ruff check + mypy + pytest
```
