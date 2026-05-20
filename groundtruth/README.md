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
4. **Has a title** — a 245 data field with a non-empty subfield `a`.

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

## Usage

```bash
cd groundtruth
pdm install
pdm run pd-groundtruth acquire --out-dir data/candidates [--manifest-url URL] [--max-records 50000] [--max-dumps N]
```

Output shards are written as `data/candidates/candidates_00001.xml`, each a
valid `<collection>` capped at 5000 records.

## Development

```bash
cd groundtruth
pdm run gates   # ruff format + ruff check + mypy + pytest
```
