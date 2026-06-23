# Design

How `pd-matcher` is built, why, and what the algorithm is actually doing under the hood.

This document is for developers and reviewers who need to understand the codebase, the technology choices, the matching pipeline, and the underlying science. For user-facing instructions and command examples, see [README.md](../README.md). Unfamiliar statistics, matching, or tooling terms are defined in the [glossary](GLOSSARY.md).

---

## 1. Overview

The tool answers one question, at scale:

> Given a MARC bibliographic record for a book, what is the most likely matching entry (registration, and optional renewal) in the U.S. Copyright Office's Catalog of Copyright Entries?

The tool is a **linkage producer, not a public-domain determiner**: it outputs verified MARC↔CCE links and the evidence behind them. Downstream consumers apply their own copyright reasoning (Cornell's matrix, the URAA restoration rules, country-of-origin analysis) to those links.

The answer is non-trivial because:

- The MARC catalog and the CCE were produced by different institutions for different purposes, with different field conventions and different error rates.
- Hard identifiers (LCCN, ISBN) are sparse and have ~5% transcription error rates — they are signals, not magic shortcuts.
- Titles, authors, and publishers drift between sources (transcription errors, abbreviation conventions, embedded edition info, OCR artifacts).
- The corpus is multilingual — Spanish, French, German, Italian, with diacritics and language-specific stopword and stemming rules.
- Years drift by 1–2 between publication, registration, and renewal.

(Whether a matched work is in the public domain — a piecewise function over publication year, registration status, renewal status, country of origin, and the current "moving wall" date — is the consumer's determination to make, not this tool's. We produce the link and surface the registration/renewal evidence it rests on.)

The codebase decomposes this into six layers, each independently testable and reviewable:

1. **Parsing** (`parsers/`) — streaming readers for MARCXML, CCE registration XML, CCE renewal TSV (the CCE files in their NYPL-transcribed form).
2. **Normalization** (`normalize/`) — Unicode/diacritic stripping, multilingual stopword removal, Snowball stemming, multilingual number/ordinal/abbreviation expansion.
3. **Indexing** (`index/`) — LMDB-backed persistent index with year-bucket blocking and a precomputed registration↔renewal join.
4. **Scoring** (`match/scorers/`) — pure-function scorers, one per signal type, each emitting structured `Evidence`.
5. **Combination + calibration** (`match/combiners/`) — weighted-mean (default, uncalibrated) and learned (LightGBM, self-calibrated) combiners, plus an optional Platt probability calibrator for the weighted mean.
6. **Parallel execution** (`workers/`) — spawn-based multiprocessing harness with producer/worker/writer/reporter roles.

A typer-based CLI (`cli.py`) is the thin wrapper that wires these layers into the user-facing commands.

---

## 2. Technology decisions

Each choice below is paired with the alternative it replaced and the reason for the swap. The git history is the canonical record of when each decision landed; this section captures the broader rationale that ties them together.

### Python 3.14, standard CPython (not free-threaded)

3.14 is the newest stable CPython at project start. The free-threaded ("no-GIL", `t` suffix) build is explicitly avoided — many C extensions we depend on (`lmdb`, `lxml`, `rapidfuzz`, `PyStemmer`, `msgspec`) do not yet ship free-threaded wheels and fail to install or behave unpredictably. The interpreter is pinned via `.tool-versions` (asdf) and `requires-python = ">=3.14"` in pyproject.toml. No `.python-version` (pyenv artifact) is shipped.

### PDM for dependency and environment management

PDM was the user's choice over Poetry, uv, or pipenv. The interface is consistent (`pdm install`, `pdm add`, `pdm remove`, `pdm sync`) and the lockfile is deterministic. The `[tool.pdm.scripts]` block defines `fmt`, `lint`, `typecheck`, `test`, and a `gates` composite so a single `pdm run gates` runs all four quality checks in order. Every Python tool invocation in this project goes through `pdm run` — never bare `python`, `pytest`, `mypy`, `ruff`, or `pip`.

### msgspec for typed records and serialization

Initially we used pydantic v2. It failed our strict typing requirement: pydantic's `BaseModel.__init_subclass__(**kwargs: Unpack[ConfigDict])` triggers `disallow_any_explicit = true` at every subclass declaration because `ConfigDict` is a `TypedDict` containing `Any`-typed fields. The pydantic mypy plugin doesn't silence it. Workarounds (per-module mypy overrides, `# type: ignore`) violated project policy.

msgspec replaces it cleanly:

- `msgspec.Struct(frozen=True, forbid_unknown_fields=True)` for every record type. Frozen + slots by default — exactly what we wanted for memory and immutability.
- `msgspec.yaml.decode(...)` / `msgspec.msgpack.Encoder(type=T)` for serialization. Schema-compiled, ~10–80× faster than pydantic, no `Any` leakage.
- One library for both config validation and cross-process wire format. We dropped raw `msgpack` in favor of `msgspec.msgpack` in Phase 3.

### LMDB for the persistent index

LMDB (Lightning Memory-Mapped Database) is the index store. The choice is structural:

- **Memory-mapped reads**: the index file maps once into virtual memory. Workers don't load anything — they read from the OS page cache. With N workers and one index file, total memory is the file size (plus per-process overhead), not N × file size.
- **Lock-free multi-reader**: any number of processes can open the same LMDB env read-only with `lock=False`. No coordination, no synchronization, no contention.
- **Spawn-friendly**: works identically under `fork` and `spawn` start methods. We use `spawn` everywhere; `fork` is unsafe on macOS and increasingly avoided on Linux.
- **Crash-safe**: writes go through a COW B+tree with a single-writer lock. Index builds are atomic from a reader's perspective.
- **Cheap to inspect**: named sub-DBs let us partition data by access pattern (records by id, records by year bucket, renewals by registration key, build metadata) without rolling our own serialization format.

Five named sub-DBs in `caches/cce.lmdb`:

| Sub-DB | Key | Value |
|---|---|---|
| `reg_by_id` | NYPL UUID bytes | msgspec.msgpack(`IndexedNyplRegRecord`) |
| `ren_by_id` | NYPL UUID bytes | msgspec.msgpack(`NyplRenRecord`) |
| `reg_by_year` | 2-byte big-endian year | msgspec.msgpack(`tuple[uuid, ...]`) |
| `ren_by_oreg` | `oreg + "|" + odat.isoformat()` UTF-8 | renewal UUID bytes |
| `meta` | string keys | build timestamp, source-hash, schema-version, counts |

The year key is big-endian so LMDB's natural byte ordering matches numeric year ordering — useful if we ever want to range-scan by year.

The registration↔renewal join is **precomputed at build time**: for each registration, we look up its renewal via `make_renewal_key(reg.regnum, reg.reg_date)` and stamp `was_renewed: bool` on the indexed record. Workers never join at match time.

### rapidfuzz for fuzzy string matching

Rust-backed token-set similarity. Author and publisher scorers use `rapidfuzz.fuzz.token_set_ratio`, which is O(n + m) for two tokenized strings of length n and m. Pure-Python alternatives (like `difflib.SequenceMatcher`) are 100× slower at our corpus scale.

### PyStemmer for Snowball stemming

C-backed Snowball stemmer with bindings for every language we care about (English, French, German, Spanish, Italian). Stemmer instances are cached per language at module level — the constructor is the expensive part.

### typer + rich for the CLI

`typer` gives type-hint-driven command and option declaration. Each `pd-matcher` subcommand's signature *is* its argument parser. `rich` provides the colored help text and (in Phase 6's reporter) the progress UI when stdout is a TTY.

A subtle point: tests need to assert substrings of help output. typer's rich formatter injects ANSI escapes around individual tokens, which can split the substring (`--marc` might appear as `\x1b[1m-\x1b[22m\x1b[1m-\x1b[22m\x1b[1mmarc\x1b[22m`). We force the test CliRunner to use `NO_COLOR=1` and `TERM=dumb` so substring assertions are stable. Production CLI keeps full color.

### structlog for logging

Structured logging from day one. `bind_contextvars` lets the workers stamp `marc_id` and `worker_id` on every log line without thread-local hacks. The renderer is `structlog.dev.ConsoleRenderer` in human mode and `structlog.processors.JSONRenderer` in JSON mode (selected by the global `--json-logs` flag). For aggregation into Loki, Splunk, etc., the JSON renderer is what you want.

### Spawn-only multiprocessing

The matching pipeline uses `multiprocessing.get_context("spawn")` unconditionally. `fork` has a long history of subtle bugs on macOS (with Cocoa, GIL, and threading combinations) and is being deprecated in newer Python versions. `spawn` works identically on every platform and pairs naturally with LMDB's mmap-shared model: each worker re-imports its module set and opens the index read-only, then loops.

The exact process topology for one `run_match` invocation:

- **Main**: orchestrates everything, hosts the producer (streams MARC), hosts the reporter thread (aggregates stats).
- **N worker processes** (spawn): each opens LMDB read-only, loads the IDF table (plus the Platt calibrator if `caches/calibrator.msgpack` exists, and the LightGBM model when `--scorer learned`), then consumes batches of MARC records from the input queue.
- **1 writer process** (spawn): drains the output queue, writes JSONL (one match record per line).

Two `multiprocessing.Queue` instances thread the work:
- input queue (`maxsize = workers * 4`) — backpressure when workers fall behind
- output queue (unbounded) — writer is fast, no need to throttle

A third queue (stats, unbounded, lightweight) carries `RecordProcessed`, `WriterHeartbeat`, etc. events to the reporter thread. msgspec.msgpack is the cross-process codec — schema-compiled, no pickle.

Graceful shutdown is a single `multiprocessing.Event` checked between batches. SIGINT in main flips the event; producer stops feeding, workers drain their current batch and exit, writer flushes and closes the JSONL output, reporter prints final stats. A second SIGINT short-circuits the cleanup with a hard exit.

### ftfy for parse-time encoding hygiene

The MARCXML and NYPL CCE transcriptions carry a long tail of encoding accidents that are invisible on inspection but disastrous downstream: classic mojibake from earlier double-encoding (``Ã©`` for ``é``, ``Â©`` for ``©``), inline byte-order marks (``U+FEFF``) embedded mid-string, and bidirectional formatting characters (``U+200E``, ``U+200F``, the ``U+202A``-``U+202E`` embedding/override family) that split otherwise-identical tokens for the downstream scorers.

We delegate the heavy lifting to [ftfy](https://ftfy.readthedocs.io/), which handles mojibake repair, BOM removal, NFC normalization, and lossy-sequence replacement in a single `fix_text` call. A tiny `str.translate` postpass additionally drops the bidirectional formatting marks that ftfy preserves by design (they are semantically meaningful inside bidi-aware renderers, but in our data they only appear as transcription artifacts).

`pd_matcher.normalize.encoding.clean_text` wraps this into a `CleanedText(text, mojibake_fixed)` result. Every parser routes each finalized subfield value through `clean_text` after its own punctuation strip. Per-parser stats counters (`MarcParseStats`, `NyplRegParseStats`, `NyplRenParseStats`) carry a `mojibake_fixed_count` so dataset quality can be surfaced in run reporting without re-walking the source files.

For CCE renewals (TSV, read as raw bytes rather than parsed as XML by lxml), we additionally probe each file at open time with a whole-file UTF-8 strict decode. The supplied corpus always passes the probe, so the hot path uses `csv.reader` directly. When the probe fails — possible if a future ingest mixes a Windows-1255-encoded Hebrew slice into the corpus — the parser switches to a bytes-level reader that routes every cell through `pd_matcher.normalize.cp1255_fallback.decode_subfield`. That decoder tries strict UTF-8, then strict Windows-1255 (accepted only if the result contains at least one Hebrew-block codepoint, to reject cp1255 decodings that succeed but produce garbage), then UTF-8 with `errors="replace"`. The two fallback counters (`subfields_decoded_as_cp1255`, `subfields_decoded_with_replacement`) are zero on the current corpus and exist to make a future quality regression visible the first time it appears.

### Pre-commit + ruff + mypy strict

The pre-commit hook runs `ruff format` and `ruff check --fix` on touched files, plus the standard `end-of-file-fixer`, `trailing-whitespace`, and `check-merge-conflict` hygiene hooks. Slow gates (`mypy`, `pytest`) are deliberately not in pre-commit — they run via `pdm run gates`.

mypy is `--strict` plus `disallow_any_explicit = true`, `disallow_any_generics`, `warn_return_any`, `warn_unreachable`, `no_implicit_reexport`. There are zero per-module overrides for `disallow_any_explicit`. There are no `# type: ignore` comments anywhere in the codebase. There are no `Any` types anywhere in our code (a few `Stemmer` C-extension stubs don't ship type info; we ignore-missing-imports them and never expose their types in our API surface).

100% line + branch test coverage is enforced (`--cov-fail-under=100`, `--cov-branch`). Allowed `# pragma: no cover` cases are narrow and listed in pyproject:

- `if __name__ == "__main__":` blocks
- `raise AssertionError("unreachable")` in exhaustive enum dispatch
- Protocol method signature stubs (never executed)
- The second-SIGINT escape hatch (would tear down pytest itself if exercised)

---

## 3. The matching algorithm

The end-to-end flow for one MARC record:

```
MarcRecord
    │
    ▼
lookup.candidates_for_year(year, window=0)        # year-bucket blocking (exact year by default)
    │
    ▼
for each candidate (typically 100s, not millions):
    │
    │  ScorerContext = (lang, stopwords, stemmer, idf, config)
    │
    │  for each (scorer, marc_field, candidate_field) pairing:
    │      Evidence = scorer.score(marc_field, candidate_field, ctx)
    │
    │  combiner.combine(evidence_list)
    │      └─ weighted_mean (default):
    │           raw = Σ(weight_i × evidence_i.normalized) / Σ(weight_i)  over present evidence
    │           calibrated = raw / 100                                   uncalibrated pass-through (default)
    │           calibrated = sigmoid(a × raw + b)                        only when a Platt calibrator artifact is present
    │      └─ learned (--scorer learned): LightGBM emits a calibrated probability directly
    │
    ▼
sort candidates by calibrated desc, keep best + top-3 alternates
    │
    ▼
MatchResult(best, alternates, candidates_considered)
```

The output is the ranked linkage. Copyright status is not computed here — a downstream consumer reads `MatchResult` (the matched registration, optional renewal, and per-field evidence) and applies its own copyright reasoning.

### Step 1: Blocking by year

A naïve nested-loop pass is 2.17M registrations × M MARC records = quadratic and infeasible. The CCE index has a `reg_by_year` sub-DB mapping each year to a list of UUIDs. The matcher pulls `[year - window, year + window]` and dedupes. The default window is **0** (the exact publication year only), so for a record published in 1955 the matcher considers just that year's bucket instead of all 2.17M registrations.

The exact-year default is chosen empirically (see [studies/year-window.md](studies/year-window.md)): on the ground-truth set, window 0 beats ±1/±2/±3 on F1 and is roughly 7× faster. Publication, registration, and renewal years do drift among the three sources, but the renewal join and the year recorded in the index absorb most of that, so widening the window adds candidates (and runtime) without improving recall. The window remains configurable in `matching.yaml` for catalogs with noisier years.

### Step 2: Per-field scoring

Each scorer is a pure function `(marc_field, candidate_field, ctx) -> Evidence`. The Evidence struct:

```python
class Evidence(Struct, frozen=True):
    scorer: str                                    # "title.token_set", "name.author", ...
    score: float                                   # 0 .. max
    max: float                                     # typically 100.0
    skipped: bool                                  # input absent → skipped
    decisive: bool                                 # set by hard-id scorers (lccn/isbn) for audit
    features: tuple[tuple[str, float], ...]        # named sub-features for ML / debugging
```

Skipped is the key invariant: a scorer with missing inputs returns `Evidence(skipped=True, score=0, max=100)` instead of throwing or scoring 0. The combiner excludes skipped Evidence from both the numerator and denominator of the weighted mean — a record with no author field doesn't get penalized as if "author scored 0."

### Step 3: Configurable field pairings

Title, author, and publisher are *transposable*: a publisher records the series title where the work title belongs, or the publisher name lands in NYPL's `claimant` element instead of `publisher`. For these fields the matcher tries several `(MARC field, CCE field)` pairings, scores each through the field's scorer, and keeps the best Evidence per scorer group; the losers are preserved in `CandidateMatch.losing_evidence` so a human auditor can see "we also tried these pairings and they scored lower."

Which pairings to try is **configuration**, not code. The pairing set lives in `src/pd_matcher/config/defaults/field_pairings.yaml`, so tuning it is a config edit and a re-eval, not a code change.

#### Code surfaces raw subfields; config composes and pairs them

This is the deliberate boundary that keeps the subsystem fully typed and bounded:

- The MARC parser emits a fixed, typed `MarcRecord` of **raw subfields** (it now keeps `245$a` as `title_main` distinct from the fused `title`, and extracts `245$n`/`$p`). The CCE side already exposes `title`, `author_name`, `publisher_names`, `claimants`.
- A finite **raw-field registry** in `match/pairing_compiler.py` (`MARC_FIELDS`, `CCE_FIELDS`) exposes each raw subfield by name through an explicit, typed accessor returning `tuple[str, ...]` (scalar fields wrap to a 0/1-tuple; list fields pass through). There is no `getattr` — that would leak `Any`; every accessor is written out.
- YAML composes registry entries via a **closed combine vocabulary**: `first` (first non-empty value), `concat`/`join` (join non-empty values by a separator). That is the *entire* expressive surface — composing already-extracted subfields. Config cannot express arbitrary logic *by design*: the scoring stays in tested code, and a typo or unknown field name cannot silently produce a degraded matcher.
- `compile_pairings(cfg)` resolves every field name against the registries and every pairing against the named field maps **once, at load time**, raising `ConfigError` on any unknown name. Typos fail at startup, not silently at match time. The result is a `CompiledPairings` of plain typed callables, bucketed by scorer group, ready for the hot loop. This mirrors the established library pattern for MARC→index mapping (cf. Traject).

#### Default pairings

| Group | MARC field | CCE field | Catches |
|---|---|---|---|
| title | `title` (fused $a+$b) | `title` | the normal case |
| title | `title_main` ($a only) | `title` | $b is subtitle noise the CCE title lacks |
| title | first `series_titles` | `title` | work title stored as series |
| author | `main_author` | `author_name` | the normal case |
| author | `statement_of_responsibility` | `author_name` | no 1xx; author only in 245$c |
| author | `main_author` | `claimants` | author recorded as the claimant |
| publisher | `publisher` | `publisher_names` | the normal case |
| publisher | `publisher` | `author_name` | self-published / author-as-publisher |

The combiner keys on one Evidence per group tag (`title.token_set`, `name.author`, `name.publisher`), so best-per-group selection yields exactly one Evidence per tag — the combiner is unchanged.

The hard-signal scorers (`lccn`, `isbn`, `year`, `edition`) compare specific typed scalars, are not transposable, and stay hard-wired in the pipeline — they are deliberately *not* part of the pairing subsystem.

### Step 4: Combine + calibrate

The combiner is a plain weighted mean over **present** (non-skipped) Evidence. Default weights, summing to 1.0:

| Scorer | Weight |
|---|---|
| title | 0.35 |
| author | 0.20 |
| publisher | 0.10 |
| year | 0.10 |
| edition | 0.05 |
| lccn | 0.10 |
| extent | 0.05 |
| volume | 0.05 |
| isbn | 0.00 |

A perfect LCCN match in isolation contributes `0.10 × 100 = 10` to the raw score, not 100. **Hard identifiers do not short-circuit.** In Phase 4 we considered making `lccn.decisive = true` short-circuit to confidence 1.0; we walked that back when the user pointed out that LCCN and ISBN have ~5% transcription error rates in this corpus. The `decisive` flag is still on Evidence for audit and ML feature inspection, but the combiner is a plain weighted mean.

By default the weighted-mean combiner runs **uncalibrated**: `calibrated = raw / 100`, a linear pass-through. Mapping the raw score to a genuine probability is optional and requires a separately-fit Platt calibrator artifact at `caches/calibrator.msgpack`. When that file is present the pipeline loads it and maps each raw score through the fitted sigmoid:

```
calibrated = 1 / (1 + exp(-(a × raw + b)))
```

The artifact is **not** built during index or match — nothing in the engine fits it automatically. It is fit out-of-band by `scripts/fit_calibrator.py`, which scores the resolved labeled-vault pairs with the production combiner and partitions the raw scores into positives (`verdict=match`) and negatives (`verdict=no_match`), then fits `a` and `b` by Newton iteration. See `docs/findings/fit_calibrator_2026-06-07.md` for the most recent fit and its outcome. The published `combined_score` in the JSONL output is `calibrated × 100` (the 0–100 scale `--min-score` compares against).

### What happens to the match

The pipeline stops at the ranked `MatchResult`. There is no copyright-assessment stage in this codebase: an earlier design embedded a YAML rule engine (`copyright/`, `assess()`, `CopyrightAssessment`) that applied Cornell's matrix, but it was removed — the project is a **linkage producer, not a public-domain determiner**. The published output is the verified MARC↔CCE link plus the registration/renewal evidence behind it. A consumer decides public-domain status by reading that linkage and applying its own copyright reasoning (Cornell's matrix, the URAA restoration rules, country-of-origin analysis, the current moving wall).

One nuance that consumers should carry over: Cornell's Category 2 header is *"Works Registered **or** First Published in the US."* Foreign authors could (and frequently did) register works with the U.S. Copyright Office, and those registrations appear in the CCE. So a MARC record that matches a CCE registration was U.S.-registered, and the standard U.S.-formality rules apply regardless of where the book was first published — URAA restoration (Category 3) only applies to works that *failed* U.S. formalities, which a registered work by definition did not. This is exactly the kind of reasoning the linkage enables and deliberately leaves to the consumer.

---

## 4. The matching science

Why each scorer chooses the algorithm it does.

### IDF-weighted Jaccard over stems (title scorer)

The naïve algorithm — "compute string similarity between two titles" — fails on common-word collisions. "American History" (4M Google Books results) and "American Geography" should not score 50% similar just because they share "American". They share *the most common word in titles of American books*, which is statistically meaningless.

The fix is **inverse document frequency (IDF)**. We pre-compute, once per index, a table mapping each stem to its IDF: `log((N + 1) / (df[t] + 1)) + 1`, where `N` is the total number of titles and `df[t]` is the number of titles containing stem `t`. A stem in 1 title gets IDF ≈ log(N) ≈ 14; a stem in every title gets IDF ≈ 1. The title scorer's metric is then **IDF-weighted Jaccard**:

```
score = Σ IDF(t) for t in stems_marc ∩ stems_nypl
        ─────────────────────────────────────────  × 100
        Σ IDF(t) for t in stems_marc ∪ stems_nypl
```

A match on "Albuquerque" (IDF ≈ 12) overwhelms a match on "American" (IDF ≈ 2). False positives from common-word collisions essentially vanish.

The stems are produced by the normalization pipeline:
1. NFKD Unicode decomposition + diacritic stripping (`normalize_text`).
2. Tokenization (whitespace + punctuation collapse).
3. Number/ordinal/abbreviation expansion via per-language tables (`normalize_numbers`).
4. Stopword removal (per language, per field; "title" stopwords are different from "author" stopwords).
5. Snowball stemming (per language, with English fallback).

The choice of Jaccard over cosine: Jaccard is a set-similarity measure and doesn't reward repeated tokens. Titles rarely repeat content words, and when they do (e.g., "World, World, World!"), the repetition is usually noise.

### Token-set rapidfuzz ratio (author + publisher scorers)

Authors and publishers don't share enough volume for IDF to help (corpus-wide author counts are small per name; publishers fewer still). They do, however, suffer from token-order variation: "Smith, John H." vs "John H. Smith"; "Macmillan Co." vs "The Macmillan Company".

`rapidfuzz.fuzz.token_set_ratio` handles this:
1. Tokenize both strings, lowercase, set-ify (drop duplicates).
2. Compute the Levenshtein distance between three combinations: `intersection`, `intersection + remainder1`, `intersection + remainder2`.
3. Return `max(...) / sum_len × 100`.

In effect it rewards shared tokens while ignoring order. "Co." vs "Company" still hurts (different tokens) but normalization expands `Co.` to `company` upstream.

### Year as a soft signal (year scorer)

The blocker bounds the year delta to `± year_window`; within that window we want closer years to outweigh further ones. The year scorer is a linear penalty:

```
score = max(0, 100 - |Δyear| × 25)
```

So `Δyear = 0` → 100, `Δyear = 1` → 75, `Δyear = 2` → 50. With the shipped default `year_window = 0` (exact year, see Step 1) the only candidates the blocker admits already have `Δyear = 0`, so the scorer effectively always sees a perfect year; the penalty exists for catalogs that widen the window in `matching.yaml`.

### LCCN exact match (LCCN scorer)

The most precise signal we have, but ~5% of LCCN matches are still wrong due to transcription errors. We canonicalize both sides (strip whitespace, leading zeros) and require exact equality. On hit, we emit `Evidence(decisive=True, score=100)`. On miss, `Evidence(skipped=True)` — *not* `score=0`. An LCCN mismatch shouldn't doom an otherwise-strong heuristic match because the mismatch might be the typo.

The `decisive=True` flag is for audit and ML feature inspection. The combiner does **not** short-circuit on it.

### ISBN exact match (placeholder)

ISBNs were not widely adopted until the late 1960s; the CCE corpus ends in 1977. Most CCE records have no ISBN; our `IndexedNyplRegRecord` doesn't carry one. The scorer is present so the contract is stable but always returns `skipped=True`. If a later corpus extends into ISBN-era works, the placeholder fills in.

### Edition compatibility (edition scorer)

The most negative-signal-positive scorer. If both sides have an extractable edition number and the numbers differ (`1` vs `2`), that's a strong indication these are different works (or different printings of the same work). We score 0 and surface it. If only one side has an edition number, the scorer falls back to `token_set_ratio` so embedded edition info still contributes.

If both sides have no edition info, `skipped=True`. We don't penalize missing-on-both-sides.

### Platt scaling (optional, off by default)

The raw weighted-mean score is between 0 and 100. It's a heuristic — there's no reason a raw 75 means "75% likely to be a real match." Platt scaling is the optional step that would turn that heuristic into a probability, but the default `weighted_mean` combiner ships **without** a calibrator: with no `caches/calibrator.msgpack` present, `calibrated = raw / 100`, a linear pass-through. The rest of this section describes the calibrator you get only if you fit one.

Platt scaling fits a logistic regression to labeled `(raw score, is-match boolean)` pairs:

```
P(is_match | raw) = 1 / (1 + exp(-(a × raw + b)))
```

The training data is the **labeled vault**, not any precomputed registration↔renewal join: `scripts/fit_calibrator.py` resolves each non-`unsure` vault entry to its (MARC, CCE) pair, scores it with the production combiner, and partitions the resulting raw scores into positives (`verdict=match`) and negatives (`verdict=no_match`). A Newton iteration then solves for the `a` and `b` that maximize the log-likelihood of those labels. With enough well-separated training data the output is well-calibrated: P=0.75 means 75% of pairs predicted at 0.75 are actually matches.

A side benefit, when a calibrator is in place: thresholds become meaningful. `--min-score` is on the 0–100 calibrated scale, so `--min-score 70` means "only show me matches with at least 70% probability of being correct" — not "only show me raw scores above 70 vibe points."

In practice the most recent fit was **not** landed. The vault's negatives skew high (the labeler tends to surface borderline cases), so the sigmoid suppressed the mid-band and cost recall; see `docs/findings/fit_calibrator_2026-06-07.md` for the full numbers and the decision to keep running with the linear pass-through. The findings doc is the source of any concrete count or coefficient — they drift, so we don't pin them here.

The learned combiner (`--scorer learned`) produces a probability directly from the LightGBM model, so it needs no Platt step at all.

---

## 5. Architecture notes

A few decisions that shape the codebase even if they don't show up in any single layer.

### Pure-function scorers, structured Evidence

The single most important design call. The prior project (which this repo is a rewrite of) lost two years to a stateful scorer chain with side effects, hidden caches, and silent exception swallowing. The new architecture forbids all of it:

- Every scorer is `(field, field, ctx) -> Evidence`. No globals, no IO, no exceptions caught and ignored.
- Evidence is a frozen struct with named sub-features. A failing scorer returns `skipped=True` instead of a misleading 0.
- The combiner sees only Evidence — it doesn't know how any score was produced. Swapping the title scorer affects only `title.py` and its tests.

This is why the learned combiner is a clean drop-in: the LightGBM model reads features from the same Evidence objects the rule-based pipeline produces. No second feature pipeline to maintain.

### One class per file

Big files mix concerns. Small files force the developer to declare what they're modeling. We default to one class per file, with private helpers colocated. Files cap at ~500 lines.

### Spawn-only, LMDB-shared workers

The naïve parallelism story for Python is `multiprocessing.fork`: copy the parent's memory and pretend. It breaks on macOS (Cocoa hates fork), it breaks on free-threaded Python, and it leaks file descriptors in subtle ways.

Spawn + LMDB-mmap is the alternative. Each worker is a fresh process that opens the index read-only. The OS page cache serves their reads from a single mapped file. No memory sharing through Python — sharing through the kernel. This works identically on macOS and Linux.

The cost is per-worker startup time (~100 ms to import the codebase and open LMDB). For a multi-million-record MARC file, that's amortized to nothing.

### Streaming everywhere

No layer of the pipeline holds the full MARC file, the full CCE corpus, or the full results in memory:

- MARC parsing via `lxml.iterparse` with explicit `.clear()` per record.
- CCE parsing the same.
- Year-bucket lookups return iterators.
- The output queue is bounded by writer throughput, not result count.
- The JSONL writer streams one record per line and flushes per row.

A 100M-row MARC file is a runtime question, not a memory question.

### Pre-commit + gates split

Pre-commit runs fast on every commit. Gates run slow, manually or in CI. The split keeps the commit flow snappy while still enforcing the heavy quality checks at PR time. Both layers are versioned in `.pre-commit-config.yaml` and `pyproject.toml`, so a fresh clone gets the same checks.

### 100% line + branch coverage as a gate

Easier to hold at 100% than to climb back to it from 85%. Every new module is born with full coverage. Allowed `pragma: no cover` is narrow and reviewable. This is paired with a strong regression-eval discipline (Phase 8) — line coverage proves every line *executed*; the regression eval proves it produced the right *answer*.

### Regression baseline + local gate (Phase 8)

Line coverage proves code *ran*; it says nothing about whether the matcher still produces the *right answers*. The regression gate closes that gap by locking the current eval scores into a checked-in artifact and failing when a change degrades them.

The baseline lives at `tests/regression/baseline.json` (schema: `pd_matcher.eval.regression.Baseline`). It records the eval invocation params (vault path, MARC pool path, `year_window`), the locked per-MARC linkage precision/recall/F1, the threshold-independent AUC and average precision, the row counts (pairs evaluated by label, MARCs evaluated, MARCs with a top pick, MARCs with the correct top pick), and the tolerance. The gate (`tests/regression/test_regression.py`) re-runs the same eval against the current vault and LMDB index and calls `compare(...)`: it **fails if precision OR recall drops more than 2 percentage points below the baseline**. Improvements, and drops within tolerance, pass. F1 is reported alongside (fully determined by P and R). AUC and average precision are reported but not currently gated — informational while the negative-label corpus is small.

The gate is local-only — **CI is deferred** until the code is published. It is excluded from the default `pdm run pytest` run (and from coverage) by the `regression` marker, and it **skips gracefully** when `data/training/label_vault.jsonl`, `data/candidates/`, or `caches/cce.lmdb` is absent, so a fresh clone without the built index still passes the default suite. All of `regression.py` reaches 100% coverage through fast unit tests that fabricate `EvalReport` instances, never touching the index.

Run the gate with `pdm run regression` (re-runs the full vault-driven eval; ~10 minutes for the current ~2,000-entry vault). Refresh the baseline after an intentional pipeline change with `pdm run regression-baseline`, which reruns the eval and rewrites the JSON.

The gate alone is necessary but not sufficient — precision and recall can move by a few thousandths and conceal individual flips. The full per-branch shipping flow, including the per-pair diff against `main` that surfaces those flips, is documented in [PHASE_WORKFLOW.md](PHASE_WORKFLOW.md).

Two caveats. **AUC and average precision are reported but not gated**: locking them requires more `no_match` labels than the vault currently carries — the threshold-sweep precision numbers are noisy at small negative-sample sizes, so a strict gate would cry wolf. Bias labeling toward the `below_sample` band to firm them up. And **the eval drops entries gracefully when their MARC is missing from the pool** (data drift after a pool rebuild); a few drops per run is normal, but a large drop count signals the pool's composition shifted and is worth investigating before trusting the metrics.

### Code repo vs data repo: the training bundle is a submodule

The code repo (`jpstroop/pd-matcher`) tracks code, tests, the regression baseline, and the NYPL submodules. It does **not** track the labeled data directly. That lives in a separate repository at [`jpstroop/cce-marc-linkage`](https://github.com/jpstroop/cce-marc-linkage) — a CC0-licensed dataset repo pulled in-tree as the `data/training` submodule, pinned to a specific commit. The submodule holds exactly two files:

- `label_vault.jsonl` — the vault: every adjudicated verdict with the labeler's notes. The labeling UI writes here directly, so it is always current. This is the source of truth *and* the training labels.
- `marc.xml` — MARCXML of every MARC the vault references, regenerated by `dump-vault-marcs` (its default `--out`), so the pairs can be re-scored without the full candidate pool.

There is no reshaped/filtered table and no separate publish command: a frozen matches list is only valid for one catalog, and the vault *is* the training table.

The split is deliberate. The two repos have different audiences (developers vs. data consumers), different update cadences (per-feature vs. per-labeling-session), and different size trajectories (the code is bounded; the data grows with labeling effort). Bundling them would penalize one audience for the other's churn and bloat a code-only clone with data the matcher doesn't read.

Because the vault lives in the submodule, publishing is ordinary submodule hygiene: refresh `marc.xml` with `dump-vault-marcs`, commit + push both files inside `data/training`, then bump the parent repo's submodule pointer. The end-to-end commands are in [USER_GUIDE.md](USER_GUIDE.md#publishing-the-training-bundle).

The regression baseline is pinned to the submodule's vault commit — the locked metrics depend on specific vault contents — so a baseline regeneration and a vault publish move together. The matcher's matching path and the live labeling UI both read the vault straight from `data/training/`.

### DESIGN.md and git history as the durable design record

Every meaningful design decision in this codebase is documented in two places: this file and the git commit log. Commit messages describe when and why each piece landed; this document captures the broader rationale and ties decisions together across modules.

If a future contributor reads this file and asks "why msgspec?", the answer is here, in §3. The commit that introduced the swap carries its own rationale in its message. Two independent records make sure the reason doesn't evaporate as the codebase evolves.

---

## 6. Open questions and future work

- **Learned scorer (#4)** — *done.* The LightGBM combiner is built, validated on held-out pair-level separation (it beats the weighted mean), and selectable via `--scorer learned`; see [LEARNED_MATCHER.md](LEARNED_MATCHER.md). The weighted mean remains the zero-dependency default; promoting the learned combiner to the default is a possible future change.
- **Baselined regression eval (Phase 8)**: built — see §5, "Regression baseline + local gate". The gate is local-only for now; wiring it into CI is deferred until the code is published.
- **HTML report viewer (Phase 7 follow-up)**: a tiny static-site generator over the JSONL output that lets a human auditor click into individual matches and see the per-scorer Evidence with its sub-features. Mocked up; not yet built.
- **Country-of-origin signals for consumers**: country-of-origin and the URAA's country-specific Berne / WTO accession dates matter to a consumer applying copyright reasoning to the linkage, but the matcher does not compute them. Surfacing the relevant MARC fields (e.g. place/country of publication) more explicitly in the output could make that downstream reasoning easier; not currently done.
- **MARC field richness**: the parser currently extracts ~15 MARC fields. Adding 6xx (subject), 5xx (notes) could improve title disambiguation in the IDF-weighted Jaccard, at the cost of more noisy tokens.
