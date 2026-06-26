# Renewal-join recovery under registration-number normalization

Read-only measurement of how many more CCE renewals would join a registration
if the registration-number join key were normalized, instead of the raw byte
concat the index builder uses today
(`make_renewal_key(regnum, regdate)` in `src/pd_matcher/index/codec.py`).

Frozen proof: `scripts/renewal_join_recovery.py`. Parses the registration XML
tree and renewal TSV tree with the same iterators the builder uses
(`iter_nypl_reg_directory`, `iter_nypl_ren_directory`); never writes the corpus
or the index.

## Corpus

- Registrations with a regnum: 2,168,158
- Renewals: 443,693
  - with no `oreg`: 26
  - with `oreg` but no `odat`: 21

Missing `oreg`/`odat` is a non-lever: 47 renewal rows total. The unjoined
majority is not caused by missing join fields.

## Sanity gate

Reg-centric replication of the builder's join (count registrations whose raw
key hits the renewal key set) reproduces the stored `renewal_joins` exactly:
**160,239 (delta +0)**. The renewal-centric baseline below counts the inverse
direction (renewals that hit a registration key), so it differs slightly.

## Recovery table (renewal-centric)

| scheme | renewals joined | delta vs baseline |
|--------|----------------:|------------------:|
| 1 baseline (raw regnum + full date) | 162,677 | +0 |
| 2 norm regnum + full date | 168,364 | +5,687 |
| 3 norm regnum + year only | 173,946 | +11,269 |
| 4 norm regnum alone | 204,901 | +42,224 |

Lever contributions, stacked:

- regnum normalization, full date held: **+5,687** (safe)
- relaxing the date to year, on top of normalization: **+5,582** (lower risk)
- dropping the date entirely: **+30,955** (mostly unsafe — see below)

## regnum-alone collision audit

Registration numbers are **not unique**: the Copyright Office restarted the
serial numbering with the 3rd series in 1947, so the same regnum recurs across
years (documented in `data/nypl-ren/README.md`). The data confirms it:

- distinct normalized regnums: 1,221,825
- normalized regnums on >1 registration: 810,086 (66%)
- normalized regnums on >1 **distinct date**: 809,560
- max registrations sharing one normalized regnum: 143

Consequences for scheme 4 (regnum alone):

- of its 204,901 joins, 168,860 (82%) land on a regnum that maps to multiple
  distinct registration dates
- of the 30,955 renewals gained *only* by dropping the date, 15,296 (49%) land
  on such an ambiguous regnum — i.e. roughly half of the date-dropped rescues
  cannot be attributed to a single registration and are likely false merges

## Residual lever (not captured by normalization)

9,736 registration `regnum` attributes are space-separated ranges
(`A160078 A160079 A160080`). The single-valued normalizer concatenates these
into one unmatchable token; expanding them into per-number keys is a separate,
safe lever bounded by ~9,736 registrations.

## Read

The ~281k unjoined renewals are overwhelmingly **structural**, not a key-format
problem: nearly every renewal carries `oreg`+`odat`, yet most reference an
original registration that simply is not in this book-only registration corpus
(the renewal set spans all classes, sourced from the Google dataset). Key
normalization is a real but modest win: **+5,687 safely** (normalized regnum,
full date retained), rising to **+11,269** if the date is relaxed to a year
window. Dropping the date balloons the count to **+42,224**, but that lever is
unsafe — because regnums were reused after the 1947 series restart, two-thirds
of normalized regnums collide across distinct dates and about half the
date-dropped rescues land on an ambiguous key. Recommendation: normalize the
regnum in the join key and keep the date (full date, or a year window for a few
thousand more); do **not** join on regnum alone. The range-expansion lever
(9,736 registrations) is the cheapest remaining safe gain after that.
