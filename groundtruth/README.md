# pd-groundtruth

Builds the public-domain **ground-truth corpus** for [`pd-matcher`](..): a set of
human-verified `(MARC record, CCE registration)` pairs labeled **match**,
**no_match**, or **unsure**. Those labels are what we calibrate and evaluate the
matcher against.

It is a standalone PDM subproject because it carries heavier dependencies
(`requests`, `fastapi`, `uvicorn`) and a relaxed coverage bar, so its
configuration never touches the core's strict `pyproject.toml`.

## Background

`pd-matcher` (the parent project) determines whether a U.S.-published book is
**in the public domain** by matching its MARC bibliographic record against the
**U.S. Copyright Office's Catalog of Copyright Entries (CCE)**: roughly, a book
published 1923–1977 is PD if it was either not registered for copyright, or
registered but never renewed at the 28-year mark. Cornell's "Copyright Term and
the Public Domain" chart codifies the full rule set.

The matcher's pipeline is fuzzy by necessity — titles get transcribed differently
across the two corpora, authors get truncated, OCR garbles characters, and the
CCE's renewal records are partial. To know how *good* the matcher's calls are
(and to improve them), we need a labeled corpus: pairs of
`(MARC record, CCE registration)` where a human has confirmed **match**,
**no_match**, or **unsure**. That labeled corpus is the **ground truth**, and
producing it is what this subproject exists to do.

Who reads the labels:

- **The matcher itself** — for calibrating score thresholds (where does
  confidence become reliable?) and measuring precision/recall on each release.
- **The future learned scorer** — a gradient-boosted model that will replace
  the hand-tuned scoring weights. It needs both positive examples (matches) and
  hard negatives (high-scoring `no_match` pairs) to train.
- **The published PD dataset** — a downstream artifact filtered from the vault
  (`verdict == "match"`), aimed at libraries, archives, and digitization
  programs that need to know which titles in their collections are PD.

## Glossary

| term | meaning |
|---|---|
| **MARC** | MAchine-Readable Cataloging — the dominant library bibliographic format. Princeton's `bibdata` exports their catalog as **MARCXML**. A "record" describes one bibliographic item (book, score, recording, …); fields are numbered (245 = title, 008 = fixed-length metadata, etc.). |
| **CCE** | Catalog of Copyright Entries — the U.S. Copyright Office's published register of copyright registrations and renewals, 1891–1977. The matching authority for pre-1978 U.S. copyright status. |
| **NYPL** | The New York Public Library transcribed the CCE volumes into structured XML/TSV. We consume their transcriptions, not the original LoC PDFs. |
| **LMDB** | Lightning Memory-Mapped Database — the on-disk key-value store the matcher uses for its CCE index. Fast random reads across multiple worker processes; built once by `pd-matcher index build`. |
| **Moving wall** | The lower bound on publication years we care about. Anything ≤ `today.year − 95` is already PD by age (no Cornell branch to evaluate), so the wall advances every January 1. As of 2026, the wall is 1931. |
| **Cornell categories** | The Cornell "Copyright Term" chart's PD-status rows. The matcher cares mostly about **Category 2** (U.S. works published 1923–1977 without notice or registration) and **Category 3** (registered but not renewed). |
| **Confidence band** | A discrete bucket the matcher assigns to each score: `ge90` (≥ 0.90), `b80_90`, `b70_80`, `below` (< 0.70). Labels stratify across bands so we don't only label easy high-confidence pairs. |
| **Stratified sample** | A sampling scheme that takes a *fixed* number of items from each `(language, band)` cell, instead of sampling proportionally. Spreads labeling effort across the score range and across all languages. |
| **Pair** | One `(MARC record, CCE registration)` candidate to be labeled. The matcher proposes; the human disposes. |
| **Vault** | `label_vault.jsonl` — the durable, git-tracked, append-only record of every human verdict ever rendered. Source of truth for the ground truth. |

## Commands at a glance

| command | purpose |
|---|---|
| `pdm run pd-groundtruth acquire …` | Download Princeton's MARC dumps and filter to books in scope (language, year, format). Writes MARCXML shards. |
| `pdm run pd-groundtruth build-queue …` | Run the matcher on the filtered shards, stratified-sample the resulting pairs, and write a `review.db`. Auto-includes every in-pool vault label. |
| `pdm run pd-groundtruth review …` | Serve the local web UI for labeling pairs. Every verdict writes both to `review.db` and to the vault. |
| `pdm run pd-groundtruth seed-vault …` | One-shot migration: dump every current label from a pre-existing `review.db` into the vault. Idempotent. |
| `pdm run pd-groundtruth vault-into-queue …` | Recovery tool: backfill an existing `review.db` with vault entries that aren't already present. Rarely needed after the build-queue carryover fix; see "Recovery" below. |

The labeling workflow itself — what each verdict means, when to use each
reason chip, how to handle translations and reprints — is in
[`LABELING_GUIDE.md`](LABELING_GUIDE.md).

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
renewed is the signal we care about). Next to the renewal flag the card shows
the **matcher's predicted Cornell status** (the Phase 5 rule engine's verdict
for the pair) as a colored chip: green for any `PD_*` status, red for any
`IN_COPYRIGHT_*` status, grey for unknown / unresolved. When a registration
was renewed, a **renewal-details** sub-block also appears with the renewal
date, the claimants as transcribed on the renewal (with a warning marker
when they differ from the registration's claimants), any new-matter the
renewal claimed, and the renewal's id / oreg as compact metadata. The CCE
panel shows author place, claimant flag, edition, publication places,
physical description, new-matter-claimed, copies, notes, copyright date,
affidavit date, notice date, the LCCN (linked to lccn.loc.gov for
cross-referencing), and any previous registration numbers — every field
the parser was able to extract from the CCE source.

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

**Spot-check the training set** at `…/labels`: a flat table of every labeled
pair (most recent first), with the verdict, reason chips, language, and
relative time. Filters in the header narrow by `verdict`, `language`, `reason`,
or a free-text substring (`?q=…`); 100 rows per page. Click any `pair_id` to
jump to `/pair/{id}` and re-label. This is the tool for catching systematic
mistakes — for example, scanning all `match` verdicts to see if any landed on
records that shouldn't have been labeled.

**For the decision rules** (when to call something `match` vs `no_match` vs
`unsure`, how translations and reprints work, what each reason chip means),
read [`LABELING_GUIDE.md`](LABELING_GUIDE.md). The guide is short and
opinionated; consult it the first time you sit down to label and any time a
new edge case appears.

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
- **`build-queue` always carries the vault forward.** At the start of the
  run, every current vault entry is resolved against the candidate pool and
  the LMDB index, scored with the matcher's per-pair routine, and queued for
  unconditional insertion into the new `review.db` with the original verdict
  pre-applied (preserving `labeled_at`). Vault MARCs are then *excluded* from
  the per-language reservoir so the matcher doesn't propose a competing pair
  for the same record. The result: a rebuilt queue carries every persistable
  vault verdict regardless of `--sample-per-lang`. The end-of-run summary
  reads `vault pre-applied: P pairs (of M resolved); Q non-vault pairs
  queued`. A vault entry whose MARC is no longer in the pool, or whose CCE
  is no longer in the index, is logged with a WARNING and skipped for this
  build — the vault file itself is never modified, so a future build can
  pick it up if the underlying data returns. Use the back-arrow to revisit
  and re-label if needed.

### Data model

**`review.db`** (SQLite) — three tables:

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
             cce_predicted_status,
             cce_renewal_id, cce_renewal_oreg, cce_renewal_rdat,
             cce_renewal_author, cce_renewal_title,
             cce_renewal_claimants, cce_renewal_new_matter,
             evidence_json, created_at)
label       (id PK, pair_id FK→review_pair, verdict, reason, note,
             labeled_at)
label_reason (label_id FK→label, code, PRIMARY KEY (label_id, code))
```

Notes:

- `review_pair.marc_json` is the lossless serialized `MarcRecord` from the
  matcher; the denormalized `marc_*` columns are convenience copies for cheap
  list-rendering.
- `evidence_json` is the per-scorer `{name: normalized_score}` map the matcher
  produced for that pair (drives the evidence bars in the card).
- `label` is **append-only**. Re-labeling a pair inserts a new row; the
  "current" verdict for a pair is the one with the largest `label.id` for that
  `pair_id` (or equivalently, the latest `labeled_at` with a tie-break on
  insertion order). `label.reason` (singular TEXT) is a legacy column kept for
  compatibility; reasons are read from `label_reason`.
- `label_reason` is a normalized many-to-one: zero or more codes per label.

**`label_vault.jsonl`** — one JSON object per line, append-only:

```json
{
  "schema": 1,
  "marc_control_id": "9912345678906421",
  "nypl_uuid": "129B8D87-6CB2-1014-A20E-B9D6251C946A",
  "verdict": "match",
  "reasons": ["pub_differs"],
  "note": null,
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

- `schema` — integer; bumped only on breaking shape changes. Older lines stay
  valid forever (append-only).
- `(marc_control_id, nypl_uuid)` — the natural key. Multiple entries with the
  same key represent a re-label history; the **last** line wins as the current
  verdict for that pair.
- `reasons` — empty tuple for `match`; zero-or-more controlled codes otherwise
  (see [`LABELING_GUIDE.md`](LABELING_GUIDE.md) for the vocabulary).
- `labeler` — string identifier of who labeled (today, always `"jpstroop"`;
  reserved for future multi-reviewer setups).
- `marc_identifiers` — durable IDs captured at label time so the published
  matches dataset can cross-walk to LCCN / OCLC / ISBN later.

The vault is **the** source of truth. `review.db` is a queryable, derivable
working copy that the labeling app needs for fast pair-by-pair access; the
vault is what survives every rebuild.

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

After the build-queue carryover fix (jpstroop/pd-matcher#33), this command is
rarely needed in normal operation — it remains available as a recovery tool.
Use it when an existing `review.db` is missing vault entries it should
contain, for example because the queue was built with vault carryover
disabled, or because the vault file was modified out of band after the build.

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

## Common workflows

Recipes for situations that come up in practice. Each is a short, self-contained
sequence; nothing here is required reading.

### First-time setup

Build a fresh CCE index, acquire MARC, build the queue, and start labeling.

```bash
# from the repo root: build the CCE index once
pdm run pd-matcher index build --out caches/nypl.lmdb \
    --reg-dir data/nypl-reg/xml --ren-dir data/nypl-ren/data

# from groundtruth/
pdm install
pdm run pd-groundtruth acquire     --out-dir data/candidates
pdm run pd-groundtruth build-queue --pool data/candidates \
                                   --index ../caches/nypl.lmdb \
                                   --out  data/review.db
pdm run pd-groundtruth review      --db   data/review.db
```

Open <http://127.0.0.1:8000> and start labeling.

### Resume labeling tomorrow

Nothing special; the DB and vault hold all state.

```bash
pdm run pd-groundtruth review --db data/review.db
```

### Rebuild the candidate set after an upstream change

When the e-book filter changes, Princeton publishes a new bibdata snapshot,
the moving wall advances, or the matcher's scoring changes — pull fresh MARC
and rebuild the queue. The vault auto-carries-over your existing labels.

```bash
rm -rf data/candidates
pdm run pd-groundtruth acquire     --out-dir data/candidates
pdm run pd-groundtruth build-queue --pool   data/candidates \
                                   --index  ../caches/nypl.lmdb \
                                   --out    data/review.db \
                                   --rebuild
pdm run pd-groundtruth review      --db     data/review.db
```

`--rebuild` is required when `data/review.db` already contains pairs;
`build-queue` refuses to silently append (which is how 28 e-book records
contaminated a previous queue). Use `--append` if you actually want the old
behavior.

### Spot-check the training set for a systematic mistake

Open <http://127.0.0.1:8000/labels>. Filter the table by verdict
(`?verdict=match`), language (`?language=eng`), or reason
(`?reason=diff_work`). Use the `?q=` substring search to find by title or
control ID. Click any `pair_id` to jump into `/pair/{id}` and re-label.

### Purge a bad subset of labels from the vault

If you discover you've labeled records that violate scope (e.g., an entire
class of records that should have been filtered out at `acquire` time and
slipped through), the vault is the source of truth and needs cleaning. Pattern:

1. Identify the offending vault keys `(marc_control_id, nypl_uuid)`. Often
   you can derive them from `review.db` via the `marc_json.extent` or another
   denormalized column.
2. Archive the vault before mutating it.
3. Rewrite the vault, keeping only entries whose key is **not** in the offender
   set.
4. Rebuild the queue (`build-queue --rebuild`); vault carryover applies only
   the surviving labels.

There is no built-in `vault prune` subcommand today; the operation is
intentionally manual because purges are rare and you should look at what
you're removing. A worked example lives in the commit message of the
"vault: purge 28 e-book entries; rebuild from clean 129" commit.

### Revisit a single previously labeled pair

Navigate to `/pair/{id}` (or click the pair ID from the `/labels` table).
The card looks the same as a fresh one; labeling it again appends a new row
to `label` (and a new line to the vault). The "current" verdict is the
latest, but the history is preserved.

### Reset everything and start fresh (rare)

Deletes the labeled corpus. Don't do this unless you're sure.

```bash
rm data/review.db label_vault.jsonl
```

Followed by a fresh acquire + build-queue + review.

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
