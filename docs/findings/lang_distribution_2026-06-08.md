# Language distribution audit (issue #6)

Generated: 2026-06-08T10:36:02.865992+00:00

## Method

Tallies `MarcRecord.language_code` (MARC 008/35-37) over two corpora. The candidate pool is streamed shard-by-shard from `/Users/jstroop/workspace/public_domain/data/candidates` (69 `<lang>/*.xml` and `<lang>/*.marcxml` files) through `iter_marc_records()`. The vault is read from `/Users/jstroop/workspace/public_domain/data/label_vault.jsonl` via `current_entries()`, and its `marc_control_id` values are resolved against the candidate pool with `build_marc_index()`. The SUPPORTED set is derived from `pd_matcher.normalize.stemming._LANGUAGE_MAP.keys()` (`eng`, `fre`, `ger`, `ita`, `spa`); UNSUPPORTED codes fall back to the English stemmer in production scoring; NULL means the 008 control field's 35-37 slice was absent, blank, or all-fill-character.

Vault unresolved sidebar: 0 of 1290 vault MARCs did not resolve in the candidate pool and are excluded from the vault tally below. (Reported here for honesty, not folded into the percentages.)

## Corpus distribution

- Total records: 334954
- Supported: 334954 (100.000%)
- Unsupported: 0 (0.000%)
- Null: 0 (0.000%)

| code | count | pct | status |
| --- | ---: | ---: | --- |
| `eng` | 100000 | 29.855 | SUPPORTED |
| `ger` | 78448 | 23.421 | SUPPORTED |
| `fre` | 71867 | 21.456 | SUPPORTED |
| `spa` | 45396 | 13.553 | SUPPORTED |
| `ita` | 39243 | 11.716 | SUPPORTED |

## Vault distribution

- Total records: 1290
- Supported: 1290 (100.000%)
- Unsupported: 0 (0.000%)
- Null: 0 (0.000%)

| code | count | pct | status |
| --- | ---: | ---: | --- |
| `eng` | 1048 | 81.240 | SUPPORTED |
| `fre` | 146 | 11.318 | SUPPORTED |
| `ger` | 56 | 4.341 | SUPPORTED |
| `ita` | 20 | 1.550 | SUPPORTED |
| `spa` | 20 | 1.550 | SUPPORTED |

## Why this is 100%

The 100% headline is not an organic measurement — it is the consequence
of two upstream filters that already restrict the pipeline to the five
Snowball-supported codes:

1. `src/pd_groundtruth/acquire.py:41` — `_TARGET_LANGUAGES = ("eng",
   "fre", "ger", "spa", "ita")`. Records in any other language code
   never enter `data/candidates/` in the first place.
2. `src/pd_groundtruth/build_queue.py:74,158` — `_DEFAULT_LANGUAGE =
   "eng"`; null language codes are coerced to English at queue-build
   time. In production neither the matcher nor the labeler sees a
   null code.

So `stemmer_for()`'s English fallback at
`src/pd_matcher/normalize/stemming.py:29` is a defensive backstop, not
a live code path. It would only fire if (a) `acquire`'s
`_TARGET_LANGUAGES` was widened to include codes without a Snowball
stemmer, or (b) the matcher was repointed at an unfiltered corpus.

## Decision

- Corpus UNSUPPORTED: 0.000% (threshold 1.0%)
- Vault UNSUPPORTED: 0.000% (threshold 1.0%)

Numbers do not justify follow-up; recommend closing #6 with this finding as durable record. If `acquire`'s allow-list ever widens — e.g. to add Dutch, Portuguese, Russian, Swedish, or any other code with a Snowball stemmer available — this audit should be re-run **on a one-off basis BEFORE merging the acquire change** so we know whether to widen `stemmer_for`'s `_LANGUAGE_MAP` in the same cycle. Same applies if we ever decide to admit codes without a Snowball stemmer (Hebrew, Arabic, CJK, Cyrillic, Greek): at that point #6's underlying question — silent English fallback for non-Snowball scripts — becomes a real issue and the no-stem path discussed in the original ticket would need to be designed.
