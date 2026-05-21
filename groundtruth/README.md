# pd-groundtruth

A standalone PDM subproject that builds the public-domain **ground-truth
corpus** for [`pd-matcher`](..). It streams Princeton `bibdata` MARC dumps,
keeps only the in-scope records, and writes the survivors as lossless MARCXML
shards for a later human-review phase (Phase 2).

This project is intentionally separate from the core `pd-matcher`: it carries
heavier dependencies (`requests` now, a review UI later) and a relaxed coverage
bar, so its configuration never touches the core's strict `pyproject.toml`.

## Filter criteria

A record survives only if **all** of the following hold (decided directly off
the raw leader, 008, and 245):

1. **Monograph book** — leader position 6 is `a` and position 7 is `m`.
2. **Supported language** — 008 positions 35:38 are one of
   `eng`, `fre`, `ger`, `spa`, `ita`.
3. **Publication year** — 008 positions 7:11 parse as a 4-digit integer in the
   inclusive range `[1923, 1977]`. Unknown/partial values (`uuuu`, `||||`,
   blanks, etc.) are rejected.
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

## Per-language partitioning

Survivors are partitioned by their 008 language code into per-language
subdirectories, each with its own cap. This lets the labeled dataset be built
English-first and then extended to the harder languages. Defaults are tuned so
the common case (English) dominates while the long tail is still collected:

| Language | Flag        | Default cap |
| -------- | ----------- | ----------- |
| `eng`    | `--cap-eng` | 40000       |
| `fre`    | `--cap-fre` | 2500        |
| `ger`    | `--cap-ger` | 2500        |
| `spa`    | `--cap-spa` | 2500        |
| `ita`    | `--cap-ita` | 2500        |

Each eligible record is routed to its language's writer only while that
language is configured and below its cap. The run stops when **every**
configured language has reached its cap, or dumps are exhausted, or
`--max-dumps` is hit. Rare languages usually never fill, so in practice the run
scans every dump to gather as many of those as exist — that is intentional.

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
it.

## Usage

```bash
cd groundtruth
pdm install
pdm run pd-groundtruth acquire --out-dir data/candidates \
  [--manifest-url URL] \
  [--cap-eng 40000] [--cap-fre 2500] [--cap-ger 2500] [--cap-spa 2500] [--cap-ita 2500] \
  [--max-dumps N]
```

During a run each completed dump logs a single progress line, e.g.:

```
dump done: scanned=124301 running_total=124301 eng=12500/40000 fre=831/2500 ger=1402/2500 spa=560/2500 ita=77/2500
```

with `full=[...]` appended as languages reach their caps, plus a final summary
line reporting per-language totals and the stop reason.

## Development

```bash
cd groundtruth
pdm run gates   # ruff format + ruff check + mypy + pytest
```
