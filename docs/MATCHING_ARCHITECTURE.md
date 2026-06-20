# Matching architecture: candidate retrieval vs. scoring

This document describes how the matcher finds and ranks CCE registrations for a MARC record, why it is structured the way it is, and the performance work in issue #22 that established the current design. For the scoring math itself (per-field scorers, field pairings, the combiner) see [DESIGN.md](DESIGN.md); for term definitions see [GLOSSARY.md](GLOSSARY.md).

## The core idea: matching is not scoring

The matcher has **two distinct stages**, and keeping them separate is what makes it fast:

1. **Matching = candidate retrieval.** Cheaply decide *which* registrations are even worth examining — a set operation over indexes. No similarity math.
2. **Scoring = per-field similarity.** Expensive title/author/publisher/year/ edition/LCCN/ISBN comparison, run **only on the retrieved candidates**.

Conflating the two — scoring every registration in a record's year — is what made the matcher unusably slow before #22 (see [Why](#why-this-design-22)).

## The LMDB index

`index build` materialises one LMDB environment (`caches/cce.lmdb`, `schema_version = 3`) with these named sub-DBs (`index/store.py`):

| sub-DB | key → value | purpose |
|---|---|---|
| `reg_by_id` | `uuid` → `IndexedNyplRegRecord` (msgpack) | the full registration record |
| `ren_by_id` | `entry_id` → `NyplRenRecord` | the full renewal record |
| `ren_by_oreg` | `regnum\|regdate` → `entry_id` | renewal join, resolved once at build into each registration's `was_renewed` flag |
| `reg_by_year` | `year` (uint16 BE) → uuid list | **year buckets** |
| `title_index` | title token → uuid list | **inverted token index** |
| `author_index` | author token → uuid list | inverted token index |
| `publisher_index` | publisher token → uuid list | inverted token index |
| `meta` | build metadata (schema version, source hash, counts) | idempotent rebuilds |

### Year buckets

`reg_by_year` maps a registration year to the uuids registered that year. The year is `reg_year`, which is derived with a fallback chain `regDate → copyDate → pubDate` (issue #19) so *ad interim* and other registrations that lack a `<regDate>` still land in a bucket. Year matching uses a configurable window (`year_window`, default **0** = exact year — see [studies/year-window.md](studies/year-window.md)).

### Inverted token indexes

`title_index` / `author_index` / `publisher_index` are classic inverted indexes: each maps a **token** to the list of registration uuids whose corresponding field contains that token. They are what let retrieval ask "which registrations share a word with this MARC record?" without scanning everything.

**Key generation is language-independent and identical on both sides** (`index/keys.py` — the single source of truth used by both the builder and the lookup). A field value's key set is `normalize.text.tokenize(value)` minus a fixed **combined stopword set**: the union, across all five supported languages (`eng/fre/ger/spa/ita`), of that field's stopwords. Two consequences:

- A French registration and an English MARC record drop the same filler tokens (`le`, `the`, …) regardless of either side's language code, so a shared distinguishing token still collides.
- **No stemming is applied to keys.** Stemming is a per-language *scoring* concern; applying it to keys would couple the index to the stemmer and to a language we don't reliably know for CCE records.

The builder (`index/builder.py`) accumulates the postings in memory during ingestion (alongside the year buckets) and flushes them at the end. Publisher postings draw tokens from both `publisher_names` and `claimants`.

## Candidate retrieval

`index/lookup.py::candidates_for(marc, window)`:

1. **Year set** — union of `reg_by_year` uuids over `[year-window, year+window]`.
2. **Token set** — union of the posting lists for every query token: title tokens from `marc.title`, author tokens from `marc.main_author` **and** `marc.statement_of_responsibility`, publisher tokens from `marc.publisher` (using the same `index/keys.py` helpers as the builder).
3. **Candidates = year set ∩ token set.** The token side is a *union* across fields (favouring recall — sharing *any* title/author/publisher token is enough); intersecting with the year set keeps the result bounded.
4. Fetch and decode only those uuids; dedupe; yield.

Records with no publication year, or that share no token with any registration, retrieve nothing (and therefore match nothing) — by design.

### What retrieval deliberately does **not** do

- **No LCCN/ISBN short-circuit.** A prior implementation returned the LCCN-exact registration immediately. We do not: in this corpus LCCN/ISBN carry a >5% transcription/OCR error rate, so they are treated as ordinary *scored* fields, never a retrieval bypass (see [GLOSSARY.md](GLOSSARY.md) and the data-quirks note).
- **No strict intersection across fields.** We union title/author/publisher sharers rather than requiring a record to share a title token *and* an author token, so a title-vs-author transposition or a missing field doesn't drop a real match.

### "Isn't this the lossy blocking we said to avoid?"

Token retrieval *is* candidate reduction — only registrations sharing a token get scored. The concern was that it would cost recall. It does not: it is **recall-validated** against the ground truth. When #22 introduced it, recall on the 1000-row eval went **up**, not down (see [results](#results)), because restricting to token-sharers removed same-year registrations that shared nothing yet were spuriously out-ranking the true match. If a future ground-truth check shows retrieval dropping real matches (e.g. heavily garbled non-Latin titles), the fix is to widen the indexed keys, not to abandon retrieval.

## Scoring (brief)

Only the retrieved candidates are scored. Each is run through the per-field scorers (IDF-weighted title Jaccard, rapidfuzz author/publisher, year, edition, LCCN, ISBN) over the configured field pairings; a weighted-mean combiner produces a confidence; the pipeline keeps the best plus up to three alternates. None of this changed in #22 — only *which* candidates reach it. Details in [DESIGN.md](DESIGN.md).

## Why this design (#22)

**The wall.** Originally the matcher scored a record against its *entire* year bucket. Busy years hold tens of thousands of registrations (one English record hit ~31k; some years ~76k), so a single record took ~1.6–3.2 s to match — infeasible for the millions-of-records production goal, and the blocker for building the full-MARC ground-truth dataset (#21).

**The false start.** The first attempt precomputed each registration's normalized/stemmed form and stored it inside every record (per language). It preserved behaviour but barely helped *and* bloated the index to 7.2 GB. A profile showed why: ~38% of match time was **fetching and msgpack-decoding every candidate from LMDB, one at a time** — and fatter records made the decode worse. The cost was the *access pattern* (per-candidate fetch over the whole bucket), not string prep. That approach was abandoned.

**The fix.** Add inverted token indexes and only score candidates that share a token — the standard record-linkage move, and the approach the earlier `marc_pd_tool` prototype used (its `indexer.py::find_candidates`) to reach ~100–200 records/min/worker. We adopted it with two deliberate differences from that prototype: no LCCN/ISBN short-circuit, and union (not intersection) across fields for recall.

## Results

Measured on the 1000-row / seed-42 regression eval (`year_window = 0`):

| metric | before #22 (full-bucket) | after #22 (token retrieval) |
|---|---|---|
| candidates scored / record | ~76k (worst) | **~3.8k avg** |
| throughput | ~19 records/min/worker | **~109 records/min/worker** |
| precision | 0.859 | **0.967** |
| recall | 0.808 | **0.878** |
| F1 | 0.812 | **0.920** |
| index size | 1.4 GB (schema 2) | 2.4 GB (schema 3) |

Both precision and recall *rose*: cutting same-year non-token-sharers removed false matches that had been out-ranking the true one. The regression baseline (`tests/regression/baseline.json`) was refreshed to the new numbers.

## Known limitations / follow-ups

- **Nondeterministic candidate ordering.** Retrieval builds a `set` of string uuids; iteration order varies across processes (hash randomization), so when two candidates tie on score the chosen "best" can differ, giving ~0.5 pp run-to-run jitter in the eval (well within the 2 pp regression tolerance). Making the candidate order deterministic would make the eval fully reproducible.
- **Common-token tails.** A title made of very common tokens can still retrieve a large set (the ~31k outlier). IDF-aware token selection on the query side (retrieve on rare tokens first) would trim the tail if single-record latency ever matters.
- **Recall on full / non-English records.** The recall validation above is on the *thin*, English-dominated legacy ground truth. Once the new full-MARC GT (#21) exists, retrieval recall should be re-measured per language and the indexed keys tuned if anything is being missed.
