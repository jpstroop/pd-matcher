# Glossary

Plain-language definitions of the statistics, matching, and software terms used across this project, with notes on how each applies here. Grouped into [Matching & statistics](#matching--statistics), [Software & tooling](#software--tooling), and [Domain & data](#domain--data), followed by [Further reading](#further-reading).

---

## Matching & statistics

**Ground truth.** The set of known-correct answers we measure against — here, the hand-labeled MARC↔CCE verdicts in `data/training/label_vault.jsonl` (~2,000 pairs as of mid-2026, growing as labeling continues). The matcher never sees it while matching; we only use it to score how well the matcher did.

**True positive / false positive / false negative.** For a yes/no decision: a true positive (TP) is a correct "yes" (we matched a record to the right CCE entry); a false positive (FP) is a wrong "yes" (we matched it to the wrong entry); a false negative (FN) is a missed "yes" (the right entry existed but we didn't find it).

**Precision.** Of the matches the system *reported*, the fraction that are correct: `TP / (TP + FP)`. "When it says match, how often is it right?" High precision means few false alarms.

**Recall.** Of the matches that *actually exist*, the fraction the system *found*: `TP / (TP + FN)`. "Of the real matches, how many did it catch?" High recall means few misses.

**Precision/recall trade-off.** Tightening a system (being more cautious about declaring a match) usually raises precision but lowers recall, and vice versa. The year-window study is a concrete example: exact-year matching raised precision ~4 points while costing ~0.5 point of recall.

**F1 (F1 score).** A single number combining precision and recall — their *harmonic mean*: `2 × (P × R) / (P + R)`. It rewards doing both well; a system that aces one and bombs the other scores poorly. We use it as the headline match-quality metric.

**Harmonic mean.** An average that leans toward the smaller of the inputs. For `P = 1.0` and `R = 0.5`, the ordinary (arithmetic) mean is 0.75 but the harmonic mean is ~0.67 — it refuses to let a great precision paper over mediocre recall. That's why it's the right average for F1.

**Confusion matrix.** A table cross-tabulating predicted vs. actual categories, so you can see *where* errors land, not just how many. The eval produces one over copyright-status categories (e.g. how often a "renewed / in copyright" was predicted as "not renewed / public domain").

**ROC curve (Receiver Operating Characteristic).** A plot of true-positive rate (recall) against false-positive rate as you sweep the decision threshold from very tight to very loose. A perfect classifier hugs the top-left corner; a random one tracks the diagonal. The shape shows how well the model *separates* matches from non-matches across all thresholds, independent of where you'd actually set the cut.

**AUC / AUC-ROC.** The area under the ROC curve — a single number summarizing threshold-independent ranking quality. 1.0 is perfect separation, 0.5 is no better than chance. We report it as `auc_roc` in `tests/regression/baseline.json`. The unqualified term "AUC" is conventionally read as ROC-AUC, but it's ambiguous in the wider literature: PR-AUC (the area under the precision-recall curve) exists too. This project always means ROC-AUC when it says AUC.

**Average precision (AP).** The area under the *precision-recall* curve, computed as a weighted mean of precisions at each recall level. Like AUC-ROC, it summarizes ranking quality across all thresholds, but it's more informative when positives are scarce relative to negatives (our setting — most MARC records have no CCE match). 1.0 is perfect; the value of a random ranking equals the positive class's base rate. We report it as `average_precision` in `tests/regression/baseline.json`.

**Baseline.** A frozen reference measurement. Future changes are compared against it; if a metric drops by more than a set tolerance, that's a regression. Issue #3 is about locking in a baseline so accidental quality drops fail CI.

**Regression.** Here, "quality regression" — a metric getting *worse* than the baseline. (Unrelated to "statistical regression," the curve-fitting technique.)

**Calibration / Platt scaling.** A raw match score (say 0–100) is just a heuristic; a "75" doesn't inherently mean "75% likely correct." Calibration fits a function that converts raw scores into honest probabilities, learned from the ground truth. *Platt scaling* is the specific technique we use: fit a logistic (S-shaped) curve mapping raw score → probability. After calibration, a reported 0.75 really does mean "about 75% of pairs scored this way are true matches."

**Logistic function / sigmoid.** The S-shaped curve `1 / (1 + e^-x)` that squashes any number into the range 0–1. The building block of Platt scaling and of logistic regression.

**IDF (inverse document frequency).** A weight that makes rare words count more than common ones. If a word appears in almost every title ("history"), matching on it means little; if it appears in very few ("Albuquerque"), matching on it is strong evidence. Computed roughly as `log(total titles / titles containing the word)`. Our title scorer weights shared words by IDF so common-word overlaps don't create false matches.

**Jaccard similarity.** A set-overlap measure: `|shared| / |combined|` — the count of items in both sets divided by the count in either. For titles, the "sets" are the words; two titles sharing most of their words score near 1. We use an IDF-weighted variant so rare shared words count for more.

**Tokenization / token.** Splitting text into units (usually words). "A token" is one such unit. Most scorers operate on tokens rather than raw characters.

**Token-set ratio.** A fuzzy-matching score (from the `rapidfuzz` library) that compares two strings as *sets* of tokens, ignoring word order and duplicates. It handles "Smith, John" vs. "John Smith" gracefully. Used for author and publisher fields.

**Levenshtein (edit) distance.** The minimum number of single-character insertions, deletions, or substitutions to turn one string into another. "kitten" → "sitten" → "sittin" → "sitting" is distance 3. It underlies the fuzzy ratios `rapidfuzz` computes.

**Fuzzy matching.** Approximate string matching that tolerates small differences (typos, abbreviations, reordering), as opposed to exact equality.

**Stemming.** Reducing words to a common root so variants match: "running", "runs", "ran" → "run". We use the *Snowball* stemmer, with language-specific rules. Lets "poems" match "poem" without an exact-string match.

**Stopwords.** Extremely common words ("the", "of", "and") removed before matching because they carry little distinguishing information. The lists are language-specific.

**Text normalization.** Standardizing text so trivial variations compare equal: lowercasing, stripping accents/diacritics, collapsing punctuation and whitespace. "Café, La" and "cafe la" normalize to the same thing.

**NFKD / Unicode normalization.** A canonical way of decomposing accented characters (é → e + combining-accent), which then lets us strip the accent to get a plain-ASCII form for matching. NFKD is one of Unicode's standard normalization forms.

**Mojibake.** Text garbled by decoding bytes with the wrong character encoding — e.g. "café" showing up as "cafÃ©" because UTF-8 bytes were read as Latin-1. The encoding-hygiene pass repairs these at parse time.

**Bidi (bidirectional) marks.** Invisible Unicode control characters that set text direction for right-to-left scripts (Hebrew, Arabic). They're meaningful to a renderer but appear as junk tokens to a matcher, so we strip them.

**Blocking.** In record-matching, the technique of comparing each input only against a small *candidate* subset instead of the whole other dataset, to avoid an infeasible all-pairs comparison. We block by year: a record is only compared to CCE entries from the same year (see the year-window study).

**Candidate / candidate pool.** The subset of CCE records considered as possible matches for one MARC record, after blocking.

**Record linkage / entity matching.** The broader field concerned with deciding when records in different datasets refer to the same real-world thing. This whole project is a record-linkage problem.

**Scorer.** A small, pure function that compares one field (e.g. title) of a MARC record and a CCE record and returns a similarity signal.

**Evidence.** The structured result a scorer returns — its score plus metadata (was the field present, etc.). The pipeline collects Evidence from all scorers.

**Combiner.** The component that merges all the per-field Evidence into one overall match score. Two are available: the default `weighted_mean` (a weighted average over the per-field scores, optionally Platt-calibrated) and `learned` (a LightGBM gradient-boosted classifier that emits a calibrated probability directly, selected with `--scorer learned`).

**Pairing.** A decision about *which* MARC field is compared against *which* CCE field. Most are obvious (title↔title), but some are cross-field (a MARC series title vs. a CCE title), because the data is sometimes transposed.

**Field pairing.** The configurable subsystem (Issue #1) that declares the set of `(MARC field, CCE field)` pairs the matcher tries for each transposable scorer group (title, author, publisher). The pairings live in `config/defaults/field_pairings.yaml`; the pipeline scores every pairing in a group and keeps the best Evidence. Tuning the set is a config edit, not a code change.

**Transposition.** The data quirk that motivates field pairing: a value lands in the "wrong" field across the two sources — the work title recorded as a series title, the publisher recorded as the copyright claimant, the author present only in the 245 statement of responsibility rather than a 1xx author field. Field pairings recover the signal by also comparing the cross-field combinations.

**Combine op.** One operation from the closed vocabulary a `FieldSpec` may use to compose raw subfields into a single string: `first` (first non-empty value) or `concat`/`join` (non-empty values joined by a separator). The vocabulary is finite by design so configuration composes data but cannot express arbitrary logic — that stays in tested code.

**Raw-field registry.** The finite, fully-typed map (`MARC_FIELDS`, `CCE_FIELDS` in `match/pairing_compiler.py`) from a raw subfield name to an explicit accessor returning its value(s). It is the *only* surface configuration can name; an unknown name fails at load time. Using explicit accessors instead of `getattr` keeps the code free of `Any`.

---

## Software & tooling

**Gate / quality gate.** An automated check that must pass before work counts as done. This project's gates are: type checking (mypy), lint + format (ruff), and the test suite at 100% coverage. "Gates green" = all passed.

**Test coverage — line vs. branch.** *Line coverage* asks whether each line of code ran during the tests. *Branch coverage* is stricter: for every decision point (`if`/`else`), it asks whether *both* outcomes were exercised. We require 100% of both. Coverage proves code *ran*, not that it's *correct* — that's what the eval is for.

**CI (continuous integration).** Automated checks that run on each change (eventually, on each pull request) to catch regressions before they land.

**Pre-commit hook.** Scripts git runs automatically when you commit. Ours run the fast checks (formatting, whitespace); the slow gates (types, tests) run separately.

**Type checking / mypy / strict / `Any`.** *Type checking* statically verifies that values are used consistently with their declared types, catching a class of bugs before runtime. *mypy* is the checker; *strict* mode turns on its most demanding rules. `Any` is the escape hatch that disables checking for a value — this project forbids it.

**Linter / formatter / ruff.** A *linter* flags suspicious or non-idiomatic code; a *formatter* rewrites code to a consistent style automatically. *ruff* is the fast tool we use for both.

**LMDB / memory-mapped / mmap / page cache.** LMDB is an embedded key-value database stored in a single file. It's *memory-mapped* (`mmap`): the file is mapped into the process's address space so reads look like memory access, and the operating system's *page cache* keeps hot pages in RAM — shared across all worker processes, so the index is loaded once no matter how many workers read it. This is why our matching scales across cores cheaply.

**spawn vs. fork.** Two ways Python starts worker processes. *fork* clones the parent process (fast, but fragile with threads and on macOS). *spawn* starts a fresh process (slightly slower to start, but robust and cross-platform). We use spawn everywhere; combined with LMDB's mmap sharing, workers re-open the index rather than inheriting it.

**msgspec / Struct.** A fast Python library for typed data and serialization. A `Struct` is its record type — immutable, memory-efficient, and strictly typed. We use Structs for every record and config object, and msgspec's encoder to store records in LMDB.

**PDM.** The Python dependency and environment manager for this project. Every Python command runs through it (`pdm run …`) so it uses the project's locked dependency versions.

**Submodule (git).** A git repository nested inside another at a pinned commit. The CCE registration and renewal datasets are pulled in as NYPL-transcribed submodules — versioned references to NYPL's own repositories rather than copies checked into ours.

**Snowball.** A framework and family of stemming algorithms (the successor to the classic Porter stemmer), with rule sets per language. Accessed via the `PyStemmer` library.

**ftfy ("fixes text for you").** A library that detects and repairs mojibake and other text-encoding damage. Used in the parse-time encoding-hygiene pass.

---

## Domain & data

**MARC / MARCXML.** MARC (MAchine-Readable Cataloging) is the standard record format libraries use for bibliographic data; MARCXML is its XML encoding. The input records we match are MARCXML.

**CCE (Catalog of Copyright Entries).** The U.S. Copyright Office's published catalog of copyright registrations and renewals. NYPL transcribed it into the XML/TSV datasets we match against.

**Registration vs. renewal.** Under the relevant U.S. copyright regime, a work was first *registered*; copyright then had to be *renewed* after an initial term to stay in force. A registered-but-not-renewed work generally fell into the public domain — the central signal this tool exploits.

**Public domain.** Creative works not (or no longer) under copyright, free to use without permission.

**Moving wall.** The rolling cutoff for public-domain-by-age: works older than a fixed span (here, the current year minus 95) are public domain. It advances by one year every January 1 — hence "moving."

**URAA.** The Uruguay Round Agreements Act (1994), which restored U.S. copyright to certain foreign works that had fallen into the public domain on technicalities. It only applies to works that failed U.S. formalities — which a CCE-registered work, by definition, did not.

**LCCN.** Library of Congress Control Number — a stable identifier the Library of Congress assigns to a bibliographic record. More durable than a catalog system's internal ID.

**MMS ID.** The internal bibliographic record identifier in Alma (a library services platform). Unlike an LCCN, it can change when records are reloaded — which is why the ground truth's MMS IDs no longer resolve.

**880 field.** A MARC field that holds an alternate-script representation of another field, linked to it — e.g. the original Hebrew or CJK form of a title whose main entry is romanized. Subject of issue #5.

---

## Further reading

On the matching and statistics:

- C. D. Manning, P. Raghavan, H. Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008 — TF-IDF, vector similarity, text normalization. The standard introductory text; freely available from the authors.
- P. Christen, *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection*, Springer, 2012 — blocking, candidate generation, match scoring.
- J. Platt, "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods," *Advances in Large Margin Classifiers*, 1999 — the calibration method.
- P. Jaccard, "The distribution of the flora in the alpine zone," *New Phytologist*, 1912 — the original set-similarity coefficient.
- V. Levenshtein, "Binary codes capable of correcting deletions, insertions, and reversals," 1966 — edit distance.

On the copyright domain:

- P. B. Hirtle, "Copyright Term and the Public Domain in the United States," Cornell University Library — the decision chart consumers apply to the verified linkage (this project produces the links, not the copyright determination): <https://guides.library.cornell.edu/copyright/publicdomain>

On the tools:

- Snowball stemmers: <https://snowballstem.org>
- ftfy: <https://ftfy.readthedocs.io>
