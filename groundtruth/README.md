# pd-groundtruth

A standalone PDM subproject that builds the public-domain **ground-truth
corpus** for [`pd-matcher`](..). It streams Princeton `bibdata` MARC dumps,
keeps only the in-scope records, runs them through the matcher to assemble a
stratified review queue, and serves a keyboard-driven web UI where a human
labels each `(MARC, CCE-candidate)` pair. Those labels **are** the ground truth.

The workflow is three commands, run in order:

1. **`acquire`** — pull + filter Princeton MARC dumps into MARCXML shards.
2. **`build-queue`** — match the shards against the CCE index and stratify the
   results into a `review.db` SQLite queue.
3. **`review`** — launch the local web UI and label the queued pairs.

This project is intentionally separate from the core `pd-matcher`: it carries
heavier dependencies (`requests`, `fastapi`, `uvicorn`) and a relaxed coverage
bar, so its configuration never touches the core's strict `pyproject.toml`.

## Filter criteria

A record survives only if **all** of the following hold (decided directly off
the raw leader, 008, and 245):

1. **Monograph book** — leader position 6 is `a` and position 7 is `m`.
2. **Supported language** — 008 positions 35:38 are one of
   `eng`, `fre`, `ger`, `spa`, `ita`.
3. **Publication year** — 008 positions 7:11 parse as a 4-digit integer in the
   inclusive range `[min_year, 1977]`. Unknown/partial values (`uuuu`, `||||`,
   blanks, etc.) are rejected. The **lower bound is the moving wall**, not a
   fixed year: it defaults to `today.year - 95` (= 1931 as of 2026) and is
   recomputed on every run, so it advances each January 1. Works published at or
   before the wall are already public domain by age and carry no copyright-status
   signal for a matching dataset, so keeping them would only dilute the corpus.
   The upper bound stays `1977` (the last CCE renewal year of interest). Pass
   `--min-year` to override the default (e.g. for reproducible runs).
4. **Not a government publication** — 008 position 28 must be blank (`" "`) or
   `"|"`. Any coded value (`a c f i l m o s u z` …) is dropped. U.S. (and other)
   government works are public domain by statute and were never registered in
   the Catalog of Copyright Entries, so they are pure noise for a CCE-matching
   ground truth — an early live run found ~95% of survivors were government
   publications, drowning out the records we actually need.
5. **Has a title** — a 245 data field with a non-empty subfield `a`.

Records are matched by element *local name*, so both the MARC21 slim namespace
(`http://www.loc.gov/MARC21/slim`) and the no-namespace serialization work.

## Disk streaming

Acquisition never materializes a full dump in memory or on disk beyond a single
compressed archive:

- each dump is streamed to a temporary `.tar.gz`, its md5 verified against the
  manifest, then **deleted** before the next dump;
- the archive is opened with `tarfile.open(mode="r|gz")` (streaming) and its
  single member's file object is fed straight to `lxml.etree.iterparse`;
- every record is `clear()`-ed after inspection, bounding memory to roughly one
  record at a time.

Disk peak is therefore ~one compressed dump file.

## Per-(language, decade) quotas

Survivors are partitioned by their 008 language code into per-language
subdirectories (output layout unchanged — see below), but **sampling is
constrained per (language, decade)** rather than by a flat per-language cap.
Each eligible record's decade bucket is `(year // 10) * 10`, so the buckets are
`1930, 1940, 1950, 1960, 1970` (the `1930` bucket only holds `1931–1939` given
the moving wall, and `1970` holds `1970–1977`). The bucket set is derived from
`min_year`..`1977`, so it stays correct as the wall moves.

A single `--per-decade-cap` (default **20000**) applies to every
`(target language, decade)` pair. An eligible record is kept only while its own
`(language, decade)` bucket is below the quota; otherwise it is skipped. This
prevents any one decade from dominating a language's slice and avoids
overweighting the high-volume English mid-century years.

The five target languages are `eng`, `fre`, `ger`, `spa`, `ita`. The run stops
when **every** `(language, decade)` bucket has reached the quota, or dumps are
exhausted, or `--max-dumps` is hit. The non-English buckets essentially never
fill, so in practice the run scans every dump — that is intentional: it gathers
every available non-English book and as many English-per-decade as exist.

## Output layout

Shards are written under `out_dir/<lang>/`, each a valid `<collection>` capped
at 5000 records:

```
data/candidates/
  eng/candidates_00001.xml
  eng/candidates_00002.xml
  fre/candidates_00001.xml
  ger/candidates_00001.xml
  spa/candidates_00001.xml
  ita/candidates_00001.xml
```

A language subdirectory is created only when at least one record is written to
it. The decade is a **sampling constraint only** — there are no decade
subdirectories; records of all decades for a language are interleaved across
that language's shards.

## Usage

```bash
cd groundtruth
pdm install
pdm run pd-groundtruth acquire --out-dir data/candidates \
  [--manifest-url URL] \
  [--per-decade-cap 20000] \
  [--min-year 1931] \
  [--max-dumps N]
```

`--min-year` defaults to the moving wall (`today.year - 95`); omit it for a
normal run and pass it only for reproducible/replay runs.

During a run each completed dump logs a single progress line showing English's
per-decade fill plus per-language totals for the rest, e.g.:

```
dump done: scanned=124301 running_total=124301 eng [1930]=4101/20000 [1940]=8800/20000 [1950]=12000/20000 [1960]=20000/20000 [1970]=20000/20000 | fre total=412 ger total=380 spa total=151 ita total=77
```

with a `bucket full: eng[1960] reached quota 20000` notice logged the first time
each `(language, decade)` bucket fills, plus a final multi-line summary table
reporting every per-language per-decade fill and the stop reason.

## Build the review queue

`build-queue` matches the acquired shards against the CCE index (the same engine
the production matcher uses) and writes a **stratified** sample of
`(MARC, CCE-candidate)` pairs into a `review.db` SQLite file — the queue the UI
serves. Stratification is per `(language, confidence-band)` so labeling effort is
spread across the score range rather than piling onto easy high-confidence pairs.

```bash
cd groundtruth
pdm run pd-groundtruth build-queue \
  --pool data/candidates \
  --index ../caches/nypl.lmdb \
  --out data/review.db \
  [--budget 2000] \
  [--sample-per-lang 1500] \
  [--workers N] \
  [--seed 42] \
  [-v | -vv]
```

`--index` points at the LMDB env produced by `pd-matcher index build`. `--budget`
scales the per-stratum caps proportionally; `--sample-per-lang` bounds the
reservoir drawn from each language directory. `-v` adds per-worker throughput
heartbeats (records/sec + ETA); `-vv` logs every match hit. On completion it
prints `records_sampled`, `records_matched`, `pairs_written`, and the per-stratum
counts.

## Review UI

`review` launches a local, keyboard-driven web UI for labeling the queued pairs.
Each label (**match / no_match / unsure**) is written straight to `review.db`;
that accumulating set of labels **is** the ground-truth corpus.

```bash
cd groundtruth
pdm run pd-groundtruth review --db data/review.db [--host 127.0.0.1] [--port 8000]
```

Then open <http://127.0.0.1:8000>. Ctrl-C stops the server; labels persist in the
database.

**Each card** shows the MARC record (left) against the proposed CCE candidate
(right), the per-field evidence bars, the overall score and confidence band, and
the **renewal flag** — the public-domain tell (a registration *not* renewed is the
signal we care about).

**Label with the keyboard** (the UI auto-advances to the next unlabeled pair, and
every keypress writes to `review.db`):

| key | verdict |
|---|---|
| `y` | match |
| `n` | no_match |
| `u` | unsure |
| `s` or space | skip (advance without labeling) |

On-screen buttons do the same.

**Focus a session** with URL filters — useful for the English-first curriculum:

- `…/?language=eng` (or `fre` / `ger` / `spa` / `ita`)
- add `&band=ge90` (or `b80_90`, `b70_80`, `below`) to drill into one confidence
  band

**Track progress** at `…/stats`: labeled vs. remaining, the match / no_match /
unsure tally, broken down per language. Revisit or re-label any specific pair at
`…/pair/{id}`.

## Development

```bash
cd groundtruth
pdm run gates   # ruff format + ruff check + mypy + pytest
```
