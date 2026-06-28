# NYPL registration↔renewal join: definitive cardinality analysis

**Date:** 2026-06-27
**Scope:** FULL corpus, no sampling — 2,168,402 registrations and 443,693 renewals.
**Proof:** `scripts/nypl_join_analysis.py` (committed). Parses with the production
iterators (`iter_nypl_reg_directory` / `iter_nypl_ren_directory`) and builds keys
with the production codec (`make_renewal_keys` → `normalize_regnum` /
`is_multi_regnum`). Nothing is reimplemented.

This analysis gates the "four-scenario" renewal pipeline, whose core assumption is:
**a renewal's `(normalized oreg + odat)` points to AT MOST ONE registration**, so a
renewal unambiguously identifies its registration.

## Methodology validation (reconciliation)

The script reproduces the builder's reg-centric join exactly:

| Metric | This analysis | Index meta | Δ |
|---|---|---|---|
| registrations parsed | 2,168,402 | 2,168,402 | 0 |
| renewals parsed | 443,693 | 443,693 | 0 |
| renewal_joins (regs with ≥1 renewal) | **167,317** | **167,317** | **+0** |

Exact reconciliation confirms the numbers below reflect production behavior.

A key fact about the production join: the key is
`normalize_regnum(regnum)|isoformat(date)`. The **registration side keys on the
strict `reg_date`** (`<regDate>` only, `None` when absent); the **renewal side keys
on `odat`**. So a renewal joins a registration only when the regnums normalize equal
**and the registration's `reg_date` equals the renewal's `odat` exactly**. Because
the plan wording ("reg-match within the odat year") is looser than the production
exact-date key, every cardinality question below is answered at **two granularities**:

- **EXACT-DATE** — the production key (date-precise).
- **YEAR-LEVEL** — `(normalized regnum, year)`; reg side on the derived `reg_year`,
  renewal side on `odat.year`. This is the "within the odat year" reading and
  rescues date-granularity mismatches.

---

## Q1 — Corpus + coverage

**Registrations (2,168,402):**
- with regnum: 2,168,158; no regnum: 244
- with strict `reg_date`: 2,142,793; **no strict `reg_date`: 25,609 (1.18%) — these
  can never exact-join a renewal**
- multi-number ("range") regnums: 9,736
- `reg_year` span **1837–1999**, distinct years 90; the mass is 1923–1977 (each year
  8.8k–120k), peaking 1974–1976 (~110k–120k/yr). Only a handful of stray records lie
  outside 1923–1977.

**Renewals (443,693):**
- no `oreg`: 26; no/unparseable `odat`: 21; lacking either (cannot join): 47
- joinable (both present): 443,646
- `odat` (original-registration) year span **1907–1991**, mass **1922–1963**
  (each ~3k–23k/yr); essentially nothing after 1963.
- `rdat` (renewal-filing) year span **1915–2001**, mass **1950–1991**.

The coverage triangle is consistent with prior findings: renewals filed ~1950–1991
renew originals registered ~1922–1963, which sits inside the registration corpus's
1923–1977 window.

---

## Q2 — Join-key cardinality, REGISTRATION side (exact-date)

`reg_key -> [reg uuids]`, ranges expanded via `make_renewal_keys`:

- distinct keys: **2,187,635** (exceeds the reg count because range regnums fan out)
- keys mapping to >1 registration (collisions): **1,978 (0.0904% of keys)**
- registrations sharing a key with a different registration: **4,246 (0.20% of regs)**
- max regs per key: **79**; p99 = 1; p999 = 1
- size distribution: 1→2,185,657 · 2→1,890 · 3→68 · 4→9 · 5→2 · 6–10→2 · 11–50→6 · 51–100→1

**The registration side is essentially unique per exact key** (99.91% of keys hold a
single registration). The few large keys are range/serial artifacts (below).

---

## Q3 — Join-key cardinality, RENEWAL side (exact-date)

`ren_key -> [renewal entry_ids]`:

- distinct keys: **412,908**
- keys mapping to >1 renewal: **21,402 (5.18% of keys)**
- renewals sharing a key with another renewal: **52,141 (11.7%)**
- max renewals per key: **49**; p99 = 3; p999 = 6
- size distribution: 1→391,506 · 2→16,694 · 3→2,650 · 4→1,078 · 5→487 · 6–10→436 · 11–50→57

Renewal-side multiplicity is higher than the registration side: the same original
registration is legitimately renewed in multiple renewal entries (per-volume / per-
contributor renewals). This affects the symmetric direction (Q4b), not the plan's
assumption.

---

## Q4 — THE CRITICAL ONE: join fan-out

### Q4a — per renewal, how many REGISTRATIONS does its key match?

**EXACT-DATE:**

| matches | renewals | |
|---|---|---|
| 0 | 271,291 | (no registration; scope/coverage) |
| exactly 1 | 171,933 | |
| **>1 (many-to-one)** | **422** | 2→401, 3→20, 24→1 |

- many-to-one as % of **all** renewals: **0.0951%**
- many-to-one as % of **joined** renewals (172,355): **0.2448%**
- of joined renewals, **99.755% map to exactly one registration**

**YEAR-LEVEL:**

| matches | renewals | |
|---|---|---|
| 0 | 264,905 | |
| exactly 1 | 177,657 | |
| **>1 (many-to-one)** | **1,084** | 2→1,061, 3→22, 24→1 |

- many-to-one as % of **all** renewals: **0.2443%**
- many-to-one as % of **joined** renewals (178,741): **0.6065%**
- of joined renewals, **99.39% map to exactly one registration**

Loosening to year-level rescues 6,386 more joining renewals (172,355 → 178,741) — so
date-granularity is a real but minor recall lever — while only modestly raising the
many-to-one rate (0.2448% → 0.6065%, both well under 1%).

### Q4b — per registration, how many renewals match it? (symmetric)

| renewals | registrations |
|---|---|
| 0 | 2,001,085 |
| exactly 1 | 163,423 |
| >1 | 3,894 |
| **≥1 (= renewal_joins)** | **167,317** |

Tail: 2→3,234, 3→423, 4→91, 5→36, 6→36, … up to one registration matched by 20
renewals. Multi-renewal registrations are the expected per-volume / per-contributor
renewals; this direction is many-to-one **the other way** and does not bear on the
plan.

> **Reconciliation note (Q9):** 167,317 (distinct *registrations* renewed) ≠ 172,355
> (distinct *renewals* that find a registration). The two count different sides of a
> mildly many-to-many relation: 3,894 regs match >1 renewal (lifts the renewal-side
> total) and 422 renewals match >1 reg (lifts the reg-side total). 167,317 reconciles
> exactly with the index meta; both figures are correct.

### Concrete many-to-one examples (renewal → multiple registrations)

Three structural patterns account for essentially all of them:

**1. Range/serial overlap (multi-volume law reporters).** A multi-volume whole is
registered as a range (`A777076 A777077 … A777091`) while individual volumes also
hold interior single-number registrations. The renewal cites one interior number,
which matches both the range record and the single-volume record. This cluster
(`A777xxx`, "Texas and Southwestern reporter digest" etc.) dominates the *year-level*
many-to-one examples:

```
renewal oreg='A777079' odat=1923-10-23 "KENTUCKY decisions … Southwestern reporter"
  -> reg A777076…A777091 (1923-10-23) "Texas and Southwestern reporter digest …"
  -> reg A777079        (1923-10-23) "Kentucky decisions … Southwestern reporter"
```

**2. Composite-work split registrations (same regnum + same date, different facets).**
A book and its illustrations / introduction / translation / editorial apparatus are
registered as separate `<copyrightEntry>` records under one number and date:

```
renewal oreg='A1004741' odat=1927-08-06 "Travelers' tales … Illus. by William Siegel"
  -> reg A1004741 "Travellers' tales; a book of marvels"
  -> reg A1004741 "[Illustrations] by William Siegel in the book entitled Travellers' tales…"

renewal oreg='A1018160' odat=1927-11-11 "The life and death of King John. Editing, notes…"
  -> reg A1018160 "… The life and death of King John"
  -> reg A1018160 "… [Editing, notes, appendices and index] by Stanley T. Williams …"
```

These are facets of **the same underlying work** — benign for copyright-status
purposes.

**3. True duplicate registrations (same work registered twice).**

```
renewal oreg='AF12761' odat=1925-01-01 "PIRANDELLO … Quaderni di Serafino Gubbio"
  -> reg AF12761 "Quaderni di Serafino Gubbio operatore, romanzo."
  -> reg AF12761 "Quaderni di Serafino Gubbio operatore, romano."   (near-identical)
```

One extreme outlier renewal matched **24** registrations both exact-date and year-
level (a range mega-registration).

---

## Q5 — regnum-within-year uniqueness

Over normalized regnums that carry a `reg_year` (1,218,160 distinct):

- `(regnum, year)` groups holding >1 registration: **6,429**
- registrations inside such within-year-duplicate groups: **13,127 (0.61% of regs)**

So `(normalized regnum, year)` is **unique for 99.4% of registrations**. Adding the
full date helps only marginally: the exact-date key still has 1,978 colliding keys
(4,246 regs, Q2) — i.e. the residual collisions are same-number **same-date**
composite/duplicate records (patterns 2 and 3 above), which the date cannot separate.
The date's job is not to break within-year duplicates; it is to break **cross-year
serial reuse** (Q6).

## Q6 — Serial-number reuse (why the date is required)

- normalized regnums appearing in **>1 distinct year**: **823,181 (67.58%)**
- max distinct years for one regnum: 10

Two-thirds of registration numbers are **reused across years** (the `A452577 = 1960 &
1973` pattern is the norm, not the exception). This is exactly why the join key must
carry the date — a regnum alone is hopelessly ambiguous across the corpus. But within
a single year (Q5) reuse collapses to 0.61%, so `(regnum, year)` is already nearly
unique.

## Q7 — Range regnums

- registrations with multi-number regnums: **9,736 (0.45%)**
- Range expansion is the reason `distinct reg keys (2,187,635) > registrations
  (2,168,402)`.
- Expansion **does** create cross-registration collisions: when a range record and an
  interior single-number record share a number+date, both land on the same key (Q4a
  pattern 1). These are the dominant source of *year-level* many-to-one, but they are
  still a tiny absolute count (the `A777xxx`-style clusters), and the renewal's own
  title disambiguates which volume is meant.

## Q8 — Scope mismatch (the unjoined renewals)

61.1% of renewals (271,291) match no registration exact-date; 59.7% (264,905) match
none even year-level. This is **not** primarily an out-of-coverage-year problem:

- Renewal `odat` years are overwhelmingly 1922–1963, which lies **inside** the
  registration corpus's 1923–1977 window.
- Year-level matching recovers only 6,386 of the exact-date non-joins (271,291 →
  264,905), so date-granularity explains only a sliver of the gap.

The bulk of the non-join is therefore **in-coverage-year-but-absent**: the renewal
corpus (Google/Stanford, *all* CCE classes) is far broader than NYPL's **book-focused**
registration transcription. Renewals for non-book classes, and books NYPL did not
transcribe, have no registration to join — by construction, not by a join defect.

---

## VERDICT

**The four-scenario plan's one-registration-per-renewal assumption HOLDS.**

Among renewals that find any registration:

- **EXACT-DATE: 99.755% map to exactly one registration; 0.2448% are many-to-one (422 renewals).**
- **YEAR-LEVEL ("within the odat year"): 99.39% map to exactly one; 0.6065% are many-to-one (1,084 renewals).**

Under either granularity the many-to-one rate is **well under 1%**, so "a renewal's
`(normalized oreg + odat)` identifies its registration" is sound as the pipeline's
backbone. The rare exceptions are structurally benign:

- ~half are composite-work splits or true duplicates — the multiple registrations are
  the **same underlying work**, so picking any one (or projecting all) is harmless for
  copyright-status purposes;
- the rest are range/interior-volume overlaps, where the renewal's own title cleanly
  selects the right volume.

### Recommendations

1. **Build the four-scenario pipeline.** Many-to-one is not common enough to make the
   assumption ambiguous; do not treat it as a blocker.
2. **Use year-level (or date-with-year-fallback) matching, not strict exact-date.**
   Exact-date silently drops 25,609 registrations with no `<regDate>` and loses ~6,386
   joinable renewals to date-granularity. Year-level recovers them while keeping many-
   to-one under 0.61%. This also matches the plan's "within the odat year" wording.
3. **Add a cheap tie-break for the <0.61% many-to-one renewals:** when a key hits >1
   registration, rank by title similarity against the renewal's own title (the matcher
   already computes this), or project all matched registrations when they are the same
   work (composite/duplicate case). No new infrastructure required.
4. **Keep the date in the join key.** 67.58% serial reuse across years means a regnum-
   only join is invalid; the year (or date) is mandatory.
```
