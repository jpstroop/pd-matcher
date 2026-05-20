# Design

How `pd-matcher` is built, why, and what the algorithm is actually doing under the hood.

This document is for developers and reviewers who need to understand the codebase, the technology choices, the matching pipeline, and the underlying science. For user-facing instructions and command examples, see [README.md](../README.md). Unfamiliar statistics, matching, or tooling terms are defined in the [glossary](glossary.md).

---

## 1. Overview

The tool answers one question, at scale:

> Given a MARC bibliographic record for a book, what is the most likely matching entry in the U.S. Copyright Office's Catalog of Copyright Entries, and what is the resulting public-domain status under U.S. law as of today?

The answer is non-trivial because:

- The MARC catalog and the CCE were produced by different institutions for different purposes, with different field conventions and different error rates.
- Hard identifiers (LCCN, ISBN) are sparse and have ~5% transcription error rates — they are signals, not magic shortcuts.
- Titles, authors, and publishers drift between sources (transcription errors, abbreviation conventions, embedded edition info, OCR artifacts).
- The corpus is multilingual — Spanish, French, German, Italian, with diacritics and language-specific stopword and stemming rules.
- Years drift by 1–2 between publication, registration, and renewal.
- U.S. copyright law itself is a piecewise function over publication year, registration status, renewal status, country of origin, and the current date (the "moving wall").

The codebase decomposes this into seven layers, each independently testable and reviewable:

1. **Parsing** (`parsers/`) — streaming readers for MARCXML, NYPL registration XML, NYPL renewal TSV.
2. **Normalization** (`normalize/`) — Unicode/diacritic stripping, multilingual stopword removal, Snowball stemming, multilingual number/ordinal/abbreviation expansion.
3. **Indexing** (`index/`) — LMDB-backed persistent index with year-bucket blocking and a precomputed registration↔renewal join.
4. **Scoring** (`match/scorers/`) — pure-function scorers, one per signal type, each emitting structured `Evidence`.
5. **Combination + calibration** (`match/combiners/`) — weighted-mean combiner plus a Platt-scaled probability calibrator.
6. **Copyright assessment** (`copyright/`) — typed `Facts`, predicate primitives, pragmatic-assumption wrappers, and a YAML-driven rule engine implementing Cornell's matrix.
7. **Parallel execution** (`workers/`) — spawn-based multiprocessing harness with producer/worker/writer/reporter roles.

A typer-based CLI (`cli.py`) is the thin wrapper that wires these layers into the four user-facing commands.

---

## 2. Code structure

```
src/pd_matcher/
├── cli.py                        # typer commands: index build/info, match, eval, train-scorer
├── logging_config.py             # structlog setup (JSON or console renderer)
├── models.py                     # MarcRecord, NyplRegRecord, NyplRenRecord, IndexedNyplRegRecord
│
├── config/
│   ├── schemas.py                # MatchingConfig, CopyrightRule, PredicateCall, IndexConfig (msgspec.Struct)
│   ├── loader.py                 # YAML → schema validation with clear ConfigError
│   └── defaults/
│       ├── matching.yaml         # scorer weights, year_window, min_combined_score
│       └── copyright_rules.yaml  # 14 ordered Cornell rules (Categories 2 + 3)
│
├── parsers/
│   ├── marc.py                   # lxml.iterparse streaming MARCXML reader
│   ├── nypl_reg.py               # NYPL registration XML reader
│   └── nypl_ren.py               # NYPL renewal TSV reader (handles two header schemas)
│
├── normalize/
│   ├── text.py                   # NFKD + diacritic strip + lowercase + punctuation collapse
│   ├── numbers.py                # Roman/word/ordinal → digits, multilingual abbreviation expansion
│   ├── stemming.py               # cached PyStemmer (Snowball) per language
│   ├── stopwords.py              # language-tuned stopword sets shipped as package data
│   ├── stopwords_data/           # english/french/german/spanish/italian JSON sets
│   ├── encoding.py               # ftfy-backed mojibake / BOM / bidi-mark repair (clean_text)
│   └── cp1255_fallback.py        # defensive UTF-8 / Windows-1255 / replace decode ladder
│
├── index/
│   ├── store.py                  # LMDB environment wrapper with named sub-DBs
│   ├── codec.py                  # msgspec.msgpack encoders + structured byte keys
│   ├── builder.py                # streams parsers → normalizes → writes; precomputes renewal join
│   └── lookup.py                 # read-only API: candidates_for_year, get_*, iter_registrations, stats
│
├── match/
│   ├── evidence.py               # Evidence struct (scorer, score, max, skipped, decisive, features)
│   ├── idf.py                    # one-pass IDF table over CCE titles; cached at caches/idf.msgpack
│   ├── pairings.py               # bounded field-pair permutations (title↔series, publisher↔claimants)
│   ├── pipeline.py               # match_record: MarcRecord + lookup → MatchResult
│   ├── result.py                 # CandidateMatch, MatchResult (best + top-K alternates + Evidence)
│   ├── scorers/
│   │   ├── context.py            # ScorerContext: per-record lang/stopwords/stemmer/IDF/config bundle
│   │   ├── title.py              # IDF-weighted Jaccard over stems
│   │   ├── name.py               # rapidfuzz token_set_ratio (author + publisher)
│   │   ├── year.py               # year delta as a soft signal (linear penalty)
│   │   ├── lccn.py               # exact-id match; flags decisive on hit
│   │   ├── isbn.py               # placeholder (CCE pre-dates ISBN)
│   │   └── edition.py            # edition-number compatibility (penalize explicit mismatch)
│   └── combiners/
│       ├── base.py               # Combiner Protocol + CombinedScore (raw, calibrated)
│       ├── weighted_mean.py      # plain weighted mean over present Evidence
│       ├── calibrator.py         # Platt-scaled logistic regression over (raw_score, is_match)
│       └── learned.py            # Phase 9 stub for the LightGBM combiner
│
├── copyright/
│   ├── status.py                 # CopyrightStatus StrEnum (16 leaves of the books-only matrix)
│   ├── facts.py                  # Facts: pub_year, country, language, registration flags, today
│   ├── predicates.py             # pure (facts, *args) → bool primitives; in_pd_by_age moving wall
│   ├── inference.py              # pragmatic-assumption wrappers (registered→notice, etc.)
│   ├── rules.py                  # ordered YAML rule engine; first-match-wins
│   └── assessment.py             # CopyrightAssessment (status, rule_name, explanation, assumptions)
│
├── eval/
│   └── ground_truth.py           # run_eval + EvalReport (precision/recall/F1 + status confusion)
│
├── output/
│   └── csv_writer.py             # CsvResultWriter; mirrors combined_ground_truth.csv schema
│
└── workers/
    ├── pool.py                   # run_match(...) → RunReport orchestration
    ├── producer.py               # main-thread MARC streamer + batcher
    ├── worker.py                 # per-process matcher: lookup + scorers + combiner + rule engine
    ├── writer.py                 # single writer process; drains output queue → CSV
    ├── reporter.py               # main-thread reporter; aggregates throughput, ETA, by-status
    ├── shutdown.py               # ShutdownCoordinator (SIGINT → multiprocessing.Event)
    ├── events.py                 # stats wire format (msgspec.Struct)
    └── messages.py               # output wire format (worker → writer)
```

Tests mirror this exactly under `tests/unit/`, with `tests/integration/` for cross-module smoke tests and `tests/fixtures/` for tiny hand-crafted MARC, NYPL registration, and NYPL renewal samples.

---

## 3. Technology decisions

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
- **N worker processes** (spawn): each opens LMDB read-only, loads the IDF + calibrator + ruleset, then consumes batches of MARC records from the input queue.
- **1 writer process** (spawn): drains the output queue, writes CSV.

Two `multiprocessing.Queue` instances thread the work:
- input queue (`maxsize = workers * 4`) — backpressure when workers fall behind
- output queue (unbounded) — writer is fast, no need to throttle

A third queue (stats, unbounded, lightweight) carries `RecordProcessed`, `WriterHeartbeat`, etc. events to the reporter thread. msgspec.msgpack is the cross-process codec — schema-compiled, no pickle.

Graceful shutdown is a single `multiprocessing.Event` checked between batches. SIGINT in main flips the event; producer stops feeding, workers drain their current batch and exit, writer flushes and closes the CSV, reporter prints final stats. A second SIGINT short-circuits the cleanup with a hard exit.

### ftfy for parse-time encoding hygiene

The MARCXML and NYPL CCE transcriptions carry a long tail of encoding accidents that are invisible on inspection but disastrous downstream: classic mojibake from earlier double-encoding (``Ã©`` for ``é``, ``Â©`` for ``©``), inline byte-order marks (``U+FEFF``) embedded mid-string, and bidirectional formatting characters (``U+200E``, ``U+200F``, the ``U+202A``-``U+202E`` embedding/override family) that split otherwise-identical tokens for the downstream scorers.

We delegate the heavy lifting to [ftfy](https://ftfy.readthedocs.io/), which handles mojibake repair, BOM removal, NFC normalization, and lossy-sequence replacement in a single `fix_text` call. A tiny `str.translate` postpass additionally drops the bidirectional formatting marks that ftfy preserves by design (they are semantically meaningful inside bidi-aware renderers, but in our data they only appear as transcription artifacts).

`pd_matcher.normalize.encoding.clean_text` wraps this into a `CleanedText(text, mojibake_fixed)` result. Every parser routes each finalized subfield value through `clean_text` after its own punctuation strip. Per-parser stats counters (`MarcParseStats`, `NyplRegParseStats`, `NyplRenParseStats`) carry a `mojibake_fixed_count` so dataset quality can be surfaced in run reporting without re-walking the source files.

For NYPL renewals (TSV, read as raw bytes rather than parsed as XML by lxml), we additionally probe each file at open time with a whole-file UTF-8 strict decode. The supplied corpus always passes the probe, so the hot path uses `csv.reader` directly. When the probe fails — possible if a future ingest mixes a Windows-1255-encoded Hebrew slice into the corpus — the parser switches to a bytes-level reader that routes every cell through `pd_matcher.normalize.cp1255_fallback.decode_subfield`. That decoder tries strict UTF-8, then strict Windows-1255 (accepted only if the result contains at least one Hebrew-block codepoint, to reject cp1255 decodings that succeed but produce garbage), then UTF-8 with `errors="replace"`. The two fallback counters (`subfields_decoded_as_cp1255`, `subfields_decoded_with_replacement`) are zero on the current corpus and exist to make a future quality regression visible the first time it appears.

### Pre-commit + ruff + mypy strict

The pre-commit hook runs `ruff format` and `ruff check --fix` on touched files, plus the standard `end-of-file-fixer`, `trailing-whitespace`, and `check-merge-conflict` hygiene hooks. Slow gates (`mypy`, `pytest`) are deliberately not in pre-commit — they run via `pdm run gates`.

mypy is `--strict` plus `disallow_any_explicit = true`, `disallow_any_generics`, `warn_return_any`, `warn_unreachable`, `no_implicit_reexport`. There are zero per-module overrides for `disallow_any_explicit`. There are no `# type: ignore` comments anywhere in the codebase. There are no `Any` types anywhere in our code (a few `Stemmer` C-extension stubs don't ship type info; we ignore-missing-imports them and never expose their types in our API surface).

100% line + branch test coverage is enforced (`--cov-fail-under=100`, `--cov-branch`). Allowed `# pragma: no cover` cases are narrow and listed in pyproject:

- `if __name__ == "__main__":` blocks
- `raise AssertionError("unreachable")` in exhaustive enum dispatch
- Protocol method signature stubs (never executed)
- The second-SIGINT escape hatch (would tear down pytest itself if exercised)

---

## 4. The matching algorithm

The end-to-end flow for one MARC record:

```
MarcRecord
    │
    ▼
lookup.candidates_for_year(year, window=2)        # year-bucket blocking
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
    │      └─ raw = Σ(weight_i × evidence_i.normalized) / Σ(weight_i)  over present evidence
    │      └─ calibrated = sigmoid(a × raw + b)                        Platt scaling
    │
    ▼
sort candidates by calibrated desc, keep best + top-3 alternates
    │
    ▼
MatchResult(best, alternates, candidates_considered)
    │
    ▼
build_facts(marc, MatchResult, today)
    │
    ▼
copyright.assess(facts, ruleset)
    │      └─ in_pd_by_age short-circuit first (moving wall)
    │      └─ otherwise: walk ordered rules, first match wins
    │
    ▼
CopyrightAssessment(status, matched_rule_name, explanation, assumptions)
```

### Step 1: Blocking by year

A naïve nested-loop pass is 2.17M registrations × M MARC records = quadratic and infeasible. The CCE index has a `reg_by_year` sub-DB mapping each year to a list of UUIDs. The matcher pulls `[year - window, year + window]` (default window = 2, so 5 buckets) and dedupes. For a record published in 1955, that's a few thousand candidates instead of 2.17M.

The window of ±2 is chosen empirically: publication, copyright registration, and renewal dates drift among the three sources but rarely by more than 2 years. Wider windows multiply candidate counts (and runtime) without improving recall on the ground-truth set.

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
| title | 0.40 |
| author | 0.20 |
| publisher | 0.10 |
| year | 0.10 |
| edition | 0.05 |
| lccn | 0.10 |
| isbn | 0.05 |

A perfect LCCN match in isolation contributes `0.10 × 100 = 10` to the raw score, not 100. **Hard identifiers do not short-circuit.** In Phase 4 we considered making `lccn.decisive = true` short-circuit to confidence 1.0; we walked that back when the user pointed out that LCCN and ISBN have ~5% transcription error rates in this corpus. The `decisive` flag is still on Evidence for audit and ML feature inspection, but the combiner is a plain weighted mean.

The Platt calibrator then maps the raw score to a probability:

```
calibrated = 1 / (1 + exp(-(a × raw + b)))
```

`a` and `b` are fit by Newton iteration over the project's 19,970 ground-truth positives plus 5–10× as many negatives sampled from the same year-buckets. The calibrator is trained once (during index build or first match) and persisted as `caches/calibrator.msgpack`. The published `combined_score` in the CSV is `calibrated × 100`.

### Step 5: Copyright assessment

The assessment engine consumes a `Facts` struct (built from the MarcRecord + MatchResult + today's date) and produces a `CopyrightAssessment`. The flow:

```python
def assess(facts, ruleset):
    # 1. Moving-wall short-circuit FIRST.
    if in_pd_by_age(facts.pub_year, facts.today):
        return CopyrightAssessment(status=PD_BY_AGE_PRE_95_YEARS, ...)

    # 2. Walk ordered rules. First match wins.
    for rule in ruleset.rules:
        if all_predicates_match(rule.when, facts):
            return CopyrightAssessment(
                status=CopyrightStatus[rule.then],
                matched_rule_name=rule.name,
                explanation=rule.explanation,
                assumptions=accumulated_assumptions,
            )

    # 3. Fallback.
    return CopyrightAssessment(status=UNKNOWN_NO_RULE_MATCHED, ...)
```

The moving wall is dynamic: `in_pd_by_age` returns `pub_year < today.year - 95`. As of project init (2026-05-18), that's anything published 1930 or earlier. On 2027-01-01 it'll auto-advance to "1931 or earlier" with zero code changes. Statutory year cutoffs inside the YAML rules (1931-1963, 1964-1977, etc.) are correctly static — those come from the 1976 Act and don't move.

Each rule in `copyright_rules.yaml` is a structured PredicateCall list:

```yaml
- name: registered_1931_1963_not_renewed
  when:
    - predicate: published_between
      args: [1931, 1963]
    - predicate: was_registered
    - predicate: was_renewed
      negate: true
  then: PD_REGISTERED_NOT_RENEWED
  explanation: >
    Registered work, regardless of country of first publication, that was
    not renewed during its initial 28-year term. The copyright lapsed at
    the end of that term and the work is in the public domain.
  assumptions: ["Assumed notice: registered work"]
```

Predicates are pure functions over Facts. They live in two modules:

- `predicates.py` — `(facts, *args) -> bool`. Used when the rule simply needs an objective check.
- `inference.py` — `(facts) -> tuple[bool, str | None]`. Used when the rule needs a pragmatic assumption. The returned assumption string is accumulated into the assessment's `assumptions` tuple so a human auditor can see what the engine took on faith.

Three pragmatic assumptions are wired in:

1. **Registered work → assume bore copyright notice.** The CCE registration itself implies the cataloger claimed copyright via the standard notice convention.
2. **Publisher matches GPO / agency patterns → US-government work.** A small regex tests `publisher_text` for "u.s. government printing office", "government printing office", "department of", "bureau of", etc.
3. **Foreign work pre-1923 → assume in source-country PD by 1996.** The URAA's 1996 baseline almost certainly held for any foreign work whose author was already dead 50+ years by then.

The user can disable all pragmatic assumptions with the (currently config-only) `enable_assumptions = False`, useful for sensitivity analysis.

#### Foreign-registered short-circuit

A subtle but important point: Cornell's Category 2 header is *"Works Registered **or** First Published in the US."* Foreign authors could (and frequently did) register works with the U.S. Copyright Office to obtain U.S. copyright protection — pre-Berne it was often required. Those registrations appear in the CCE.

If a MARC record matches a CCE registration, the work was U.S.-registered, and the standard U.S.-formality rules apply regardless of where the book was first published. Phase 5's first cut got this wrong (it forced `country_is_us` on every Category 2 rule, which routed matched foreign works through Category 3's URAA logic — but URAA only restores works that *failed* U.S. formalities, which a registered work by definition did not). The Phase 5 corrections branch fixed it: Category 2's registration-gated rules apply to any registered work; Category 3's URAA rules explicitly require `NOT was_registered`.

The status enum naming reflects this: `PD_REGISTERED_NOT_RENEWED` (not `PD_US_PUB_REGISTERED_NOT_RENEWED`) — it applies to a foreign-registered work just as well as a U.S.-published one.

---

## 5. The matching science

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

The blocker enforces ±2 years; within that window we want closer years to outweigh further ones. The year scorer is a linear penalty:

```
score = max(0, 100 - |Δyear| × 25)
```

So `Δyear = 0` → 100, `Δyear = 1` → 75, `Δyear = 2` → 50. Beyond 2 the blocker excludes the candidate; we never see deltas > 2 in practice.

### LCCN exact match (LCCN scorer)

The most precise signal we have, but ~5% of LCCN matches are still wrong due to transcription errors. We canonicalize both sides (strip whitespace, leading zeros) and require exact equality. On hit, we emit `Evidence(decisive=True, score=100)`. On miss, `Evidence(skipped=True)` — *not* `score=0`. An LCCN mismatch shouldn't doom an otherwise-strong heuristic match because the mismatch might be the typo.

The `decisive=True` flag is for audit and ML feature inspection. The combiner does **not** short-circuit on it.

### ISBN exact match (placeholder)

ISBNs were not widely adopted until the late 1960s; the CCE corpus ends in 1977. Most CCE records have no ISBN; our `IndexedNyplRegRecord` doesn't carry one. The scorer is present so the contract is stable but always returns `skipped=True`. If a later corpus extends into ISBN-era works, the placeholder fills in.

### Edition compatibility (edition scorer)

The most negative-signal-positive scorer. If both sides have an extractable edition number and the numbers differ (`1` vs `2`), that's a strong indication these are different works (or different printings of the same work). We score 0 and surface it. If only one side has an edition number, the scorer falls back to `token_set_ratio` so embedded edition info still contributes.

If both sides have no edition info, `skipped=True`. We don't penalize missing-on-both-sides.

### Platt scaling

The raw weighted-mean score is between 0 and 100. It's a heuristic — there's no reason a raw 75 means "75% likely to be a real match." Platt scaling fixes that.

Given a labeled training set of pairs (raw score, is-match boolean), we fit a logistic regression:

```
P(is_match | raw) = 1 / (1 + exp(-(a × raw + b)))
```

The Newton iteration solves for `a` and `b` that maximize the log-likelihood of the training labels. We use the 19,970 ground-truth positives plus sampled hard negatives (other CCE records in the same year-bucket as a positive). With sufficient training data, the calibrator output is well-calibrated: P=0.75 means 75% of pairs predicted at 0.75 are actually matches.

A side benefit: thresholds become meaningful. `--min-score 0.70` means "only show me matches with at least 70% probability of being correct" — not "only show me raw scores above 70 vibe points."

If a learned scorer (Phase 9) replaces the weighted mean, the Platt calibrator becomes unnecessary — the learned model produces a probability directly. Until then, calibration is the right place for empirical knowledge of how raw scores relate to truth.

---

## 6. Architecture notes

A few decisions that shape the codebase even if they don't show up in any single layer.

### Pure-function scorers, structured Evidence

The single most important design call. The prior project (which this repo is a rewrite of) lost two years to a stateful scorer chain with side effects, hidden caches, and silent exception swallowing. The new architecture forbids all of it:

- Every scorer is `(field, field, ctx) -> Evidence`. No globals, no IO, no exceptions caught and ignored.
- Evidence is a frozen struct with named sub-features. A failing scorer returns `skipped=True` instead of a misleading 0.
- The combiner sees only Evidence — it doesn't know how any score was produced. Swapping the title scorer affects only `title.py` and its tests.

This makes Phase 9's learned scorer a drop-in: the LightGBM model reads features from the same Evidence objects the rule-based pipeline produces. No second feature pipeline to maintain.

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
- The CSV writer is `csv.DictWriter`-based; flushes per row.

A 100M-row MARC file is a runtime question, not a memory question.

### Pre-commit + gates split

Pre-commit runs fast on every commit. Gates run slow, manually or in CI. The split keeps the commit flow snappy while still enforcing the heavy quality checks at PR time. Both layers are versioned in `.pre-commit-config.yaml` and `pyproject.toml`, so a fresh clone gets the same checks.

### 100% line + branch coverage as a gate

Easier to hold at 100% than to climb back to it from 85%. Every new module is born with full coverage. Allowed `pragma: no cover` is narrow and reviewable. This is paired with a strong regression-eval discipline (Phase 8) — line coverage proves every line *executed*; the regression eval proves it produced the right *answer*.

### Regression baseline + local gate (Phase 8)

Line coverage proves code *ran*; it says nothing about whether the matcher still produces the *right answers*. The regression gate closes that gap by locking the current eval scores into a checked-in artifact and failing when a change degrades them.

The baseline lives at `tests/regression/baseline.json` (schema: `pd_matcher.eval.regression.Baseline`). It records the eval invocation params (`--sample 1000 --seed 42 --year-window 0 --as-of 2026` over `combined_ground_truth.csv`), the locked precision/recall/F1, the row counts, and the tolerance. The gate (`tests/regression/test_regression.py`) re-runs that exact eval against the local LMDB index and calls `compare(...)`: it **fails if precision OR recall drops more than 2 percentage points below the baseline**. Improvements, and drops within tolerance, pass. F1 is reported for context but is not itself gated, since precision and recall fully determine it.

The gate is local-only — **CI is deferred** until the code is published. It is excluded from the default `pdm run pytest` run (and from coverage) by the `regression` marker, and it **skips gracefully** when `caches/nypl.lmdb` or `data/combined_ground_truth.csv` is absent, so a fresh clone without the built index still passes the default suite. All of `regression.py` reaches 100% coverage through fast unit tests that fabricate `EvalReport` instances, never touching the index.

Run the gate with `pdm run regression` (re-runs the ~1000-row eval, a few minutes). Refresh the baseline after an intentional pipeline change with `pdm run regression-baseline`, which reruns the canonical eval and rewrites the JSON.

Two caveats are recorded in the baseline's `notes`. The eval uses **thin records** reconstructed from the ground-truth CSV columns rather than full MARC, so the absolute scores are a floor, not the production ceiling. And **per-status confusion is not gated**: the ground-truth status labels predate the current `CopyrightStatus` enum, so only the precision/recall of the *match* (predicted vs. ground-truth `match_source_id`) is locked. The baseline is slated for a refresh after #19 (registration-date parsing) lands.

### design.md and git history as the durable design record

Every meaningful design decision in this codebase is documented in two places: this file and the git commit log. Commit messages describe when and why each piece landed; this document captures the broader rationale and ties decisions together across modules.

If a future contributor reads this file and asks "why msgspec?", the answer is here, in §3. The commit that introduced the swap carries its own rationale in its message. Two independent records make sure the reason doesn't evaporate as the codebase evolves.

---

## 7. Open questions and future work

- **Learned scorer (Phase 9)**: replace the weighted mean + Platt calibrator with a LightGBM model trained on the same Evidence features. Plan calls for an A/B with the current pipeline and a ≥2 F1-point threshold for adoption.
- **Baselined regression eval (Phase 8)**: built — see §6, "Regression baseline + local gate". The gate is local-only for now; wiring it into CI is deferred until the code is published.
- **HTML report viewer (Phase 7 follow-up)**: a tiny static-site generator over the CSV that lets a human auditor click into individual matches and see the per-scorer Evidence with its sub-features. Mocked up; not yet built.
- **Delayed-URAA accession dates**: Cornell's matrix references country-specific Berne / WTO accession dates that replace the 1996 baseline. Phase 5 punts these to `UNKNOWN_INSUFFICIENT_DATA`. Could be tabulated if the corpus contains enough such works to justify it.
- **MARC field richness**: the parser currently extracts ~15 MARC fields. Adding 6xx (subject), 5xx (notes) could improve title disambiguation in the IDF-weighted Jaccard, at the cost of more noisy tokens.
