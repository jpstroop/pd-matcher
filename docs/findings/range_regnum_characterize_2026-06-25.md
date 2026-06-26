# Range-regnum characterization — 2026-06-25

Read-only diagnostic. Some CCE registration `regnum` attributes carry several
numbers in one attribute (`"A160078 A160079 A160080"`). The merged
single-valued `normalize_regnum` concatenates these into one unmatchable
token, so today they neither join renewals nor expose a whole/part signal to
the matcher. This sizes two questions: (1) are these registered multi-volume
WHOLES (a whole/part signal living in the CCE data, relevant to `volume.compat`
and the issue-#82 part detector), and (2) what would per-number expansion buy
for renewal joins. No `src/` code is modified; the corpus and index are
read-only.

## Method

`scripts/range_regnum_characterize.py` streams the registration XML tree and
the renewal TSV tree with the SAME iterators the index builder uses
(`iter_nypl_reg_directory`, `iter_nypl_ren_directory`). The range detector is
the `_is_multi_regnum` helper prototyped in
`scripts/renewal_join_recovery.py` (interior whitespace AND every
whitespace-token independently shaped like a registration number, so a verbose
class phrase such as `"A ad int. 8956"` is excluded). Per-number tokens are
each run through the production `normalize_regnum`.

**Population.** Count, numbers-per-record distribution, and the
strictly-consecutive fraction (one alpha prefix, integer parts ascending by 1).

**Multi-volume signal.** Each record's own `title` / `desc` / `notes` /
`edition` is run through the production `volume.compat` detectors
(`_is_part_range`, `_is_multivolume_whole`, `_detect_part`,
`_detect_bare_designator`). "whole / covering-range" = `_is_part_range` or
`_is_multivolume_whole` (the CCE-whole direction the scorer reads); "single
part" = `_detect_part`. A uniform reservoir sample (seed 8224, n=30) plus the
first 10 signal-bearing titles are shown verbatim.

**Renewal cross-ref.** Indexing each registration under EACH number, splitting
first-number vs interior. A renewal counts as an ADDITIONAL join when it does
NOT already join the merged single-valued key set but DOES land on an
expansion key. Reported under the production-faithful full-date key and two
date-relaxed variants.

## Population

| metric | value |
|:---|---:|
| registrations parsed (with regnum) | 2,168,158 |
| multi-number regnum records | 9,736 |
| — with a reg_date (full-date join usable) | 9,468 |
| numbers per record — median | 2.0 |
| numbers per record — mean | 3.23 |
| numbers per record — max | 800 |
| strictly CONSECUTIVE runs (e.g. A160078-80) | 5,899 (60.6%) |
| arbitrary lists / gaps / mixed prefix | 3,837 (39.4%) |

Numbers-per-record distribution: `2=6,361  3=1,661  4=652  5=298  6=252
7=118  8=108  9=38  10+=248`. The mass is small ranges (65% are exactly two
numbers); the long tail (max 800) is law-report / statute serials.

## Multi-volume signal

| indicator (production `volume.compat` detectors) | records | share |
|:---|---:|---:|
| ANY volume indicator (title/desc/notes/edition) | 6,713 | 69.0% |
| whole / covering-range indicator | 6,431 | 66.1% |
| single part-designator indicator | 1,914 | 19.7% |

The whole signal dominates: two-thirds carry a `desc` like `"2 v."` / `"7 V.
… maps"` or an explicit `"v. 1-3"` / `"1.-2. bd."` range. Representative
signal-bearing titles:

- `A692774 A692775` — "History of West Virginia" — desc `"3 v. …"`
- `A692801…A692807` (7, consecutive) — "The great events of the great war" — desc `"7 V. fronts. … 24cm"`
- `AF21662 AF21663` — "Edda" — note `"… Thule … 1.-2. bd. … CONTENTS.—1. bd. …—2. bd. …"`
- `A119270…A119280` (11, consecutive) — "Indiana. Laws. statutes. etc."
- `A595400 A595401` — "Meet Soviet Russia. Book 1: … Book 2: …"
- `A945465 A945466` — "Gingivectomy/gingivoplasty … Pt.1-3."
- `A223289 A223290` — "Workbook … parts 1-2 and Part 3 …"
- `A72476 A72477` — "Callaghan's Indiana digest, complete from 1 Blackford …"
- `A696302…A696304` — "Indiana. Appellate court." — desc `"3 v. …"`
- `A279265…A279268` — "Spelling for word mastery. … Grade 5-8"

The ~31% with no detected signal are mostly two-number lists whose volume cue
lives only in untokenized desc noise (`"2 V. illus. 19½crn."`) or is genuinely
absent (a paired edition like "Bound ed. & Loose leaf ed." — two registration
acts, not a volume set). The 19.7% "single part" share overlaps the whole
share (a record can show both a `desc` count and a "Pt. 1" in the title).

## Renewal cross-ref (additional joins from per-number expansion)

Renewals parsed: 443,693 (26 with no `oreg`; only 1 with a multi-number
`oreg`, so the asymmetry is entirely on the registration side).

| scheme | any number of a range | interior (NOT first) only |
|:---|---:|---:|
| full date (production-faithful) | 3,991 | 2,409 |
| year only (date relaxed) | 4,795 | 3,076 |
| regnum alone (date dropped) | 611 | 425 |

The date-dropped row is *lower* because many interior numbers also exist as
standalone single registrations elsewhere in the catalog, so without the date
gate the renewal already "joins" one of those and is not counted as
additional. The full-date row (3,991; 2,409 on interior numbers) is the
production-faithful figure: against the ~160k existing renewal joins it is a
~2.5% lift, low-risk because the join is still date-gated on real
registration ids.

## Read

Yes — these ranges are overwhelmingly registered multi-volume WHOLES, not
arbitrary clerical groupings: 60.6% are strictly consecutive number runs and
66–69% carry an explicit whole/covering-range signal (`"N v."`, `"v.1-3"`,
`"1.-2. bd."`) in the record's own title/desc/notes — exactly the CCE-whole
direction `volume.compat` already reads. A registration covering
`A692774 A692775` whose `desc` is `"2 v."` is a registered set; a MARC that
describes one of its volumes is a true whole/part mismatch. So per-number
expansion is dual-purpose: surfacing the individual numbers would both (a)
let the volume scorer / the #82 part detector treat the regnum range as a
first-class CCE-whole signal, and (b) recover renewal joins. The renewal lift
is modest but real and cheap: 3,991 additional joins under the production
full-date key (2,409 on interior, non-first numbers), ~4,795 if relaxed to
year — a ~2.5% increase over the existing joins, date-gated and low-risk.
Both effects share the same one-time index change (index each registration
under every number of its range), which makes the lever worth doing.
