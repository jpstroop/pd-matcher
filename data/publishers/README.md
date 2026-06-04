# data/publishers/

Two hand-curated reference files describing U.S. and international book publishers active during the 1909-Act copyright window (roughly 1923–1977), plus the parent-conglomerate ancestry of the major firms that survived into the present.

| File | Purpose | Keyed by |
|---|---|---|
| [`publisher_imprints.json`](publisher_imprints.json) | Canonical names, variant spellings, imprint relationships. Consumed by the matcher's publisher scorer. | `canonical` (publisher name) |
| [`publisher_events.json`](publisher_events.json) | Structured timeline events — founded, acquired_by, merged_with, renamed, defunct — derived from the prose in `publisher_imprints.json`'s `notes` fields. Standalone research artifact; not consumed by the matcher. | Same `canonical` as above |

The two files share keys: `EVENTS.entities[<canonical>]` returns the event list for the publisher entry whose `canonical` field is `<canonical>`. 102 publisher canonicals overlap.

## Status

Working datasets, not finished ones. The schemas are stable; the publisher list is incomplete by design and grows as labeling encounters new names. The numbers below are accurate as of the most recent commit touching either file — `git log -1 data/publishers/publisher_imprints.json` is the source of truth for "when was this current."

## Scope

| | |
|---|---|
| **Era target** | 1923–1977 U.S. copyright registrations (the 1909-Act regime) |
| **Geography** | U.S. trade, paperback, university, religious + the U.K. and continental publishers that show up in U.S. registrations (Gallimard, Mondadori, Espasa-Calpe, Springer-Verlag, etc.) |
| **Out of scope** | Most modern (post-1990) imprint reshuffles; magazine houses that didn't publish books; vanity/subsidy presses |

Modern conglomerate ancestry *is* recorded when it matters for our era — e.g. Random House's 1960 acquisition of Knopf is in scope because both names appear in CCE registrations across the 1960 boundary.

## `publisher_imprints.json` — schema

One object per *canonical parent house*, each carrying its variant names, its imprints with date validity, free-text notes on lineage / ownership changes, and source citations.

```json
{
  "schema_version": 1,
  "publishers": [
    {
      "canonical": "McGraw-Hill Book Company",
      "aliases": ["McGraw-Hill", "McGraw Hill"],
      "imprints": [
        {"name": "Whittlesey House", "active": "1929-1968", "notes": "Trade-books imprint."}
      ],
      "active": "1909-present",
      "notes": "U.S. educational, professional, and (briefly) trade publisher.",
      "sources": ["https://en.wikipedia.org/wiki/McGraw_Hill", "niu:McGraw-Hill"]
    }
  ]
}
```

Field details:

- **`canonical`** — the most common modern form of the parent house's name. Every entry must have a unique canonical.
- **`aliases`** — every variant spelling, abbreviation, and era-name form that appears in real catalog records (e.g. `Doubleday, Doran` for the 1927–1946 era of Doubleday). Aliases for the *parent house only* — imprints have their own entries below.
- **`imprints[]`** — editorial sub-brands operated under the parent. `name` is the imprint's canonical form; `active` is a free-text date range (`"1929-1968"`, `"1953-present"`); `notes` is optional free text.
- **`active`** — date range for the parent house itself.
- **`notes`** — multi-line history: founding year, key ownership changes, current status.
- **`sources`** — URLs (Wikipedia preferred) or shorthand citations (`niu:McGraw-Hill` = the local NIU mirror page named McGraw-Hill.htm). Every claim in `notes` and every imprint should be traceable to a source.

A msgspec schema in `src/pd_matcher/normalize/publishers.py` enforces shape and rejects unknown fields. The CI gate decodes the bundled file on every test run, so authoring typos surface immediately.

## `publisher_events.json` — schema

A single object whose `entities` field maps publisher canonical names to a list of structured timeline events. Each event carries a typed shape and a verbatim `evidence` quote of the source phrase in `publisher_imprints.json`'s notes for auditability.

```json
{
  "schema_version": 1,
  "extracted_from": "data/publishers/publisher_imprints.json",
  "entities": {
    "McGraw-Hill Book Company": [
      {"type": "merger_formed", "year": 1909, "predecessors": ["McGraw Publishing", "Hill Publishing"],
       "evidence": "Founded 1909 by merger of McGraw Publishing and Hill Publishing."}
    ],
    "Doubleday & Company": [
      {"type": "founded", "year": 1897, "founders": ["Frank Nelson Doubleday"],
       "evidence": "Founded by Frank Nelson Doubleday."},
      {"type": "renamed", "year": 1946, "to": "Doubleday & Company",
       "evidence": "then Doubleday & Company."},
      {"type": "acquired_by", "year": 1986, "actor": "Bertelsmann",
       "evidence": "Acquired by Bertelsmann in 1986."}
    ]
  }
}
```

Event types in use today:

| `type` | Required fields | Optional fields |
|---|---|---|
| `founded` | `year` | `founders[]` |
| `merger_formed` | `year`, `predecessors[]` | — |
| `renamed` | `year`, `to` | — |
| `merged_with` | `year`, `actors[]` | — |
| `acquired_by` | `year`, `actor` | — |
| `sold_to` | `year`, `actor` | — |
| `acquired` | `year`, `target` | — |
| `defunct` | `year` | — |

Every event also carries an `evidence` string quoting the source phrase from `publisher_imprints.json`. External actors (Bertelsmann, RCA, News Corp, ...) appear as `actor` / `target` values without their own entries — they're not book publishers in their own right.

No msgspec schema enforces this file today (it's research-only). A future consumer would add one alongside the loader.

## Sources used

1. **Wikipedia** — primary. Carries internal imprint structure of U.S. trade publishers during 1923–1977 better than any other free source.
2. **`/tmp/niu_publishers/`** — a local `wget --mirror` of https://ulib.niu.edu/publishers/ (a 2007 Northern Illinois University library snapshot, not currently maintained). Useful for the late-1970s+ acquisition chronology of the 12 conglomerates it covers (Bertelsmann, Holtzbrinck, McGraw-Hill, Pearson, Reed Elsevier, Springer, Taylor & Francis, John Wiley, Kluwer, Blackwell, Thomson, Candover/Cinven). Hazard: NIU's prose contains spelling typos — verify any name against Wikipedia before adopting it verbatim (e.g. NIU writes *Whittlesley*; the correct form is *Whittlesey*).
3. **Internet Archive Wayback Machine** — used to fill 404s and missing pages on the NIU mirror.

## How the matcher uses the data

At matcher startup, `pd_matcher.normalize.publishers.get_default_alias_index()` loads `publisher_imprints.json`, normalizes every canonical / alias / imprint name, and builds a `dict[str, str]` mapping `normalized_name → normalized_canonical`. `score_publisher` (in `src/pd_matcher/match/scorers/name.py`) consults this index after its fuzzy match: when both sides of a publisher pair resolve to the same canonical, the score is lifted to at least 95.0 — preserving any higher fuzzy score under a `max` rather than blindly replacing it.

The matcher does NOT consume `publisher_events.json`. It exists purely for research / future use.

The normalizer (`normalize_publisher`) strips a small fixed set of stop-words:

> the, a, an, and, &, co, company, corp, corporation, inc, incorporated, ltd, limited, publishing, publishers, publisher, press, publications, publication, pub, books, book, editorial, editions, verlag, librairie, et, cie, sons, of

…then lowercases, drops non-alphanumerics, and collapses whitespace. The list lives verbatim in `normalize_publisher`'s module and is intentionally small — we want `McGraw-Hill Book Company, Inc.` to collapse to `mcgraw hill`, but we *don't* want generic words like `university`, `daily`, `new`, `north`, etc. stripped, because they carry distinguishing signal.

## Known limitations

- **Modern children's imprints** (Beginner Books, Margaret K. McElderry, etc.) are sparse.
- **Sheet-music publishers** beyond G. Schirmer aren't covered.
- **Foreign-language houses** lean French / German / Italian; Slavic, Asian, and Iberian coverage is thin where vault evidence is also thin.
- **Date enforcement at lookup time is not implemented.** The `active` field is recorded as a string and is human-readable, not machine-enforced. If an alias should only apply within a date window (e.g. Whittlesey House ↔ McGraw-Hill before 1968), we don't currently enforce that. Adding date-windowed alias resolution is a follow-up (tracked at jpstroop/pd-matcher#67).
- **Yield is modest.** Across an 819-MARC labeled match-verdict set, alias resolution flips +2 top picks from wrong to right and 0 from right to wrong — the bulk of "low publisher score on a labeled match" cases are *CCE storing only an author-claimant*, not alias mismatch. The table fixes what it can; the rest is a different problem.

## Adding an entry

1. **Confirm the name appears in our data.** Either it surfaced in a vault note ("Maybe an imprint?") or `gh issue` traffic flagged it. Wikipedia has thousands of publishers; only encode the ones we encounter.
2. **Verify spelling against Wikipedia.** Cross-check NIU spellings.
3. **Pick the canonical form.** The most common modern long form is usually right (`McGraw-Hill Book Company`, not `McGraw-Hill`).
4. **List variant spellings** as `aliases`. Include abbreviation forms (`McGraw Hill`, `McGraw-Hill Inc.`) and era-name forms (`Doubleday, Doran` for Doubleday).
5. **Nest imprints under the parent.** If an imprint had a significant independent era *before* its acquisition (e.g. Pantheon Books 1942–1961 before Random House bought it), give it its own top-level canonical entry *too* — the cross-reference is intentional and the code tolerates it.
6. **Cite every source.** Wikipedia URLs in full; NIU as `niu:<filename>`.
7. **Add structured events to `publisher_events.json` if the new entry's `notes` describe any.** Quote the source phrase verbatim in the `evidence` field of each event. Pick the right type from the table above; omit fields not stated in the notes (don't fabricate years or actors).

## Independent research value

The two files together capture a piece of U.S. publishing history that's surprisingly hard to find in a single machine-readable place: which imprints belong to which parent house with date ranges (`publisher_imprints.json`), plus the structured acquisition / merger / rename / defunct timeline of how the 1923–1977 landscape consolidated (`publisher_events.json`). If you're working on a different bibliographic-matching project or studying mid-20th-century publishing consolidation, these files are structured to be useful on their own — drop in your own loader against the JSON schema and ignore everything in `src/pd_matcher/`.
