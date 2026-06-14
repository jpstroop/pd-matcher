# CCE-side part-of-a-larger-work signal — 2026-06-13

Issue #82, Phase-1 prep. Sizes a CCE-side part detector (the maintainer's redirected approach) and confirms where the MARC parsed-notes designator noise comes from by tag. No `src/` code is modified; the vault is read-only.

## Method

**Question.** Issue #82's redirected approach: the cleanest whole/part signal is on the CCE side — *does this registration look like part of a larger work?* — detectable by regex over the CCE record's OWN `title` + `notes` + `desc`. A CCE record is self-referential (it describes the registered work), so a designator in its text is citation-clean, unlike a MARC 500 note that leaks volume citations to OTHER works.

**Part 1 — MARC note noise by source tag.** The parser flattens MARC tags {500, 502, 505, 520} into one `MarcRecord.notes` tuple and drops the tag, so this RE-READS the raw MARCXML (`data/candidates/<lang>/*.xml`) for every MARC in the whole/part slice via `lxml.iterparse`, keeping each note's source tag. For every note carrying a `volume._PART_NUMBER_RE` designator it records the tag; within 500 it splits citation-style notes (lead phrases `Reprinted from`, `Abstracted in`, `First published in`, `Reprint of`, `Issued as/in`, `Indexed in`, `Reviewed in`, `Translation of`, …) from contents-style notes (≥2 `v.N`/`pt.N` items = this work's own volumes).

**Part 2 — CCE part detector.** `_cce_looks_like_part(cce)` scans the CCE `title`, then each `note`, then `desc` (first firing field wins). Each field is tested by `volume._detect_part` (the existing `_PART_NUMBER_RE`) FIRST, then additive series-context patterns: `german_half` (`1. hälfte` / `1. Halbbd.`), `plural_vols` (`Vols. 3-4`), `volume_range` (`v.20-21`), `series_trailing_comma` (`, v.3` / `, Bd.40` / `, vol. VII` — a kind prefix is REQUIRED after the comma so bare `<title>, 86` numbers do not match), and `parenthetical_series` (`(The reference shelf, vol. VII, no. 6)`). `_is_multivolume_whole` is NOT consulted — a multi-volume whole is the WHOLE, not the part.

**Coverage** is the detector hit-rate over the tagged whole/part cases (`marc_whole_cce_part` / `cce_whole_marc_part`), by category and overall.

**False-positive rate.** The clean-negative pool is every vault entry with verdict `match` that is NOT tagged a whole/part category AND whose free-text `note` carries no whole/part keyword — standalone single-work matches whose CCE reg should NOT look like a part. A deterministic seeded sample (`Random(8224)`, n=200, key-sorted before and after sampling) is run through the detector; any hit is a false positive.

## Part 1 — MARC note noise by source tag

Over the 49 tagged whole/part MARC records, 38 had raw 5xx notes. Counts below are designator-bearing notes (a `_PART_NUMBER_RE` hit) by source tag, with the total note count per tag for context.

| tag | designator-bearing notes | all notes |
|:---|---:|---:|
| 500 | 7 | 35 |
| 502 | 0 | 0 |
| 505 | 18 | 28 |
| 520 | 0 | 0 |
| **all** | **25** | **63** |

### Within tag 500: citation-style vs contents-style

A 500 note opening with a citation lead phrase refers to ANOTHER work (noise). A 500/505-style note enumerating ≥2 `v.N`/`pt.N` items lists THIS work's own volumes (salvageable signal). `other` is neither — a single designator with no citation lead.

| 500 class | count |
|:---|---:|
| citation-style (noise) | 0 |
| contents-style (own volumes) | 2 |
| other (single, no lead) | 5 |

**Citation-style 500 examples (noise)**

_None in this run._

**Contents-style 500 examples (own volumes)**

- `500` Vol. 1 tr. by Alexis N. Obolensky; v. 2-3 tr. by John R. Schulenberger.
- `500` Vol. 1 has half-title: Study of the judicial system of Maryland. The Judicial Council of …

**Other 500 examples (single, no lead)**

- `500` Vol. 3- rev. and edited by Richard Wilson, from a text prepared by W.Freeman Galpin and O…
- `500` "How to use the Interpreter's Bible [by] George Arthur Buttrick." (16 p.) inserted in v. …
- `500` Vol. 16: Comprehensive index.
- `500` [Vol.6- have copyright 1946-
- `500` [Vol.6] has title: Gesammelte werke.

### Verdict — is MARC 500 salvageable?

- Tag 500 carries 28% of all designator-bearing notes in the slice.
- Within 500, 0% are citation-style (noise about another work), 29% are contents-style (this work's own volumes), 71% other.

**500 noise is not dominated by recognizable citation lead phrases.** A lead-phrase filter would leave most designator-bearing 500 notes in place, so 500 is hard to salvage by that rule alone. This strengthens the case for the CCE-side detector (Part 2) over consuming MARC 500.

## Part 2 — CCE part detector: coverage

Resolved 49 of 49 tagged whole/part cases (0 CCE missing in index).

| category | cases | flagged | coverage |
|:---|---:|---:|---:|
| `marc_whole_cce_part` | 42 | 27 | 64% |
| `cce_whole_marc_part` | 7 | 1 | 14% |
| **overall** | **49** | **28** | **57%** |

### Coverage by firing pattern

| pattern | hits |
|:---|---:|
| `base_part_number_re` | 24 |
| `german_half` | 2 |
| `parenthetical_series` | 1 |
| `plural_vols` | 1 |

### Coverage by CCE source field

| source field | hits |
|:---|---:|
| `cce.title` | 22 |
| `cce.notes` | 4 |
| `cce.desc` | 2 |

### Misses (21 cases the detector did NOT flag)

| marc_control_id | nypl_uuid | category |
|:---|:---|:---|
| 9911542063506421 | 2179C037… | marc_whole_cce_part |
| 9912106123506421 | DA0A2A92… | cce_whole_marc_part |
| 99125488430606421 | F5562F4B… | cce_whole_marc_part |
| 9916779783506421 | B0A65F23… | marc_whole_cce_part |
| 9917787763506421 | DB01768A… | cce_whole_marc_part |
| 9920104013506421 | ABBB06E4… | marc_whole_cce_part |
| 9923856593506421 | 7DA7A0F7… | marc_whole_cce_part |
| 9924200073506421 | 8C72F951… | cce_whole_marc_part |
| 9925002403506421 | 06A4FE77… | marc_whole_cce_part |
| 9926062383506421 | A24759B0… | marc_whole_cce_part |
| 9926296013506421 | 25541090… | marc_whole_cce_part |
| 9926313483506421 | 0ABE3996… | marc_whole_cce_part |
| 9927261683506421 | 577CACEF… | marc_whole_cce_part |
| 9927332983506421 | 8AD303B5… | cce_whole_marc_part |
| 9928073873506421 | 31925220… | marc_whole_cce_part |
| 9928129433506421 | A530B7EB… | marc_whole_cce_part |
| 9929738383506421 | 213614D9… | marc_whole_cce_part |
| 9929860013506421 | 8F4CB11E… | cce_whole_marc_part |
| 993265753506421 | F128BF2E… | marc_whole_cce_part |
| 994882173506421 | AF9D7F26… | marc_whole_cce_part |
| 998461223506421 | C6A01363… | marc_whole_cce_part |

## Part 2 — CCE part detector: false-positive rate

- Clean-negative pool: 917 entries (verdict `match`, untagged, keyword-free note).
- Seeded sample: 200 (`Random(8224)`).
- False positives: **14** → **FP rate 7.0%**.

### False positives by pattern

| pattern | hits |
|:---|---:|
| `base_part_number_re` | 10 |
| `series_trailing_comma` | 4 |

### False positives by CCE source field

| source field | hits |
|:---|---:|
| `cce.title` | 2 |
| `cce.notes` | 8 |
| `cce.desc` | 4 |

### False-positive examples (what tripped the regex)

| marc_control_id | nypl_uuid | pattern | field | matched text |
|:---|:---|:---|:---|:---|
| 9911941423506421 | 68ADECE3… | base_part_number_re | cce.title | A critical history of English literature. Vol.1-2. |
| 99131189301306421 | B7F28076… | base_part_number_re | cce.title | Art of the printed book 1455 - 1955 |
| 9913998173506421 | BE182F02… | base_part_number_re | cce.desc | 324 p., 1 l. 21ͨͫ. |
| 9916063613506421 | AA515FEF… | base_part_number_re | cce.notes | Die Grundlehren der mathematischen Wissenschaften, Bd.66 |
| 9917542613506421 | FC950B11… | series_trailing_comma | cce.notes | Annals of mathematics studies, no.64 |
| 9917731663506421 | 92316BF2… | base_part_number_re | cce.notes | Add. ti: Criminal justice administration: cases and materials. Successor volume to Miller… |
| 992045423506421 | C19AF149… | base_part_number_re | cce.notes | Papers of the Peabody Museum of Archaeology and Ethnology, Harvard University, v.55, no.1 |
| 9921078803506421 | A3C747D3… | base_part_number_re | cce.desc | viii, 682 p. illus., xiv col. pl., diagrs. 22½ͨͫ |
| 992411913506421 | 01728637… | base_part_number_re | cce.desc | viii p., 2 l., 468 p. illus. (maps) 24ͨͫ |
| 9927508683506421 | A190F669… | series_trailing_comma | cce.notes | Herzl Institute pamphlet, no.14 |
| 9932412313506421 | CE6BFBD2… | series_trailing_comma | cce.notes | Research monograph, no.30 |
| 9948041753506421 | CC4875A9… | base_part_number_re | cce.desc | viii. 72 p. 23½ͨͫ. |
| 9948556303506421 | 14B987C3… | series_trailing_comma | cce.notes | Industrial Relations monograph, no.25 |
| 9949153473506421 | D0A3CAAE… | base_part_number_re | cce.notes | Ergebnisse der Mathematik und ihrer Grenzgebiete, Bd.40 |

## Decision — is the CCE-side detector Phase-1's primary signal?

- **Coverage**: 57% of resolved whole/part cases flagged (28/49).
- **False-positive rate**: 7.0% (14/200 clean negatives).

**Recommendation: promising coverage but FP rate (7.0%) is above the 5% tolerance — tighten before shipping.** Inspect the false-positive examples above and demote or constrain the offending pattern(s) (the trailing-comma and parenthetical-series patterns are the usual precision risks). Re-measure FP after tightening; do NOT ship the detector as the primary signal until FP is in tolerance.

> Per the vault blind-spot finding, the labeled vault is structurally biased toward `match` pairs the matcher already surfaces; coverage here is an UPPER BOUND on top-1 flips, not a prediction. Phase 2's per-MARC diagnostic remains the real gate.
