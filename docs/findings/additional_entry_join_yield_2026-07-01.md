# additionalEntry join yield: definitive full-corpus measurement

**Date:** 2026-07-01
**Gates:** issue [#111](https://github.com/jpstroop/pd-matcher/issues/111) — should we parse
`<additionalEntry>` regnums in `nypl_reg.py` and rebuild the index?
**Scope:** FULL corpus, no sampling — 2,168,402 registrations, 443,693 renewals, 216,068
`<additionalEntry>` elements, 153 `<renewalEntry>` blocks.
**Proof:** `scripts/additional_entry_join_measure.py` (committed). Join keys are built with the
production codec (`make_renewal_keys` → `normalize_regnum` / `is_multi_regnum`); the top-level
registration side reuses the production private helpers (`_extract_reg_date`, `_text`) and the
same id/title guards as `nypl_reg._build_record`. Read-only; nothing under `data/` is written.

## The gap being measured

The production registration parser iterates only `<copyrightEntry>` and keeps a single top-level
`regnum`. The CCE guide ("Multiple claims in a single entry") documents that one `<copyrightEntry>`
can bundle several separate registrations as `<additionalEntry>` children, each with its own
`<regNum>` and (usually) `<regDate>`. Those interior numbers are dropped today, so a renewal that
cites an interior registration number cannot join to anything.

## Q1 — Methodology validation (reconciliation)

The harness reproduces the builder's reg-centric join exactly, confirming every number below
reflects production behavior:

| Metric | This analysis | Index meta | Δ |
|---|---|---|---|
| registrations parsed (guarded) | 2,168,402 | 2,168,402 | **+0** |
| renewals parsed | 443,693 | 443,693 | **+0** |
| reg-centric renewal_joins (regs with ≥1 renewal) | **167,317** | **167,317** | **+0** |

1,472 entries are skipped (no `id` or no `<title>`) exactly as production skips them. Distinct
top-level registration keys: 2,187,635 (slightly above the record count from multi-number range
fan-out).

Renewal join-field coverage: 26 renewals lack `oreg`, 21 lack `odat`; **443,646 are joinable**
(both present), spanning 412,908 distinct renewal keys.

## Q2 — additionalEntry yield: NET NEW joins (headline)

A renewal is a **net new join** when it joins NO top-level registration key but DOES join an
`<additionalEntry>`-derived key. Two additionalEntry key sets bracket the yield:

- **STRICT** — the additionalEntry's own `<regDate>` only (`None` when absent). This mirrors
  production's strict `_extract_reg_date` join key exactly. **The trustworthy floor.**
- **FALLBACK** — own `<regDate>` if present, else the parent entry's regDate (as #111 asked). The
  parent date is usually a *different* registration event, so fallback-only joins are mostly
  coincidental. **An upper bound only.**

| Renewal-centric measure | Count |
|---|---|
| joinable renewals joining a top-level key | 172,355 |
| joinable renewals currently UNJOINED | 271,291 |
| **NET NEW joins via additionalEntry (STRICT)** | **12,131** |
| net new via additionalEntry (FALLBACK) | 12,189 (**+58** over strict) |

- Strict net-new as % of currently-unjoined joinable renewals: **4.47%**
- Strict net-new as % of all joinable renewals: **2.73%**
- Strict net-new vs the current 167,317 reg-centric joins: **+7.25%**

The fallback set adds only **58** joins over strict (+0.5%), so the parent-date fallback is
worthless *and* risky — **use strict own-regDate keys only.** This is corroborated by the
inventory: only **2,189 of 216,068** additionalEntries (1.0%) lack their own `<regDate>`, so
strict already captures 99% of them.

### additionalEntry inventory

| | Count |
|---|---|
| copyrightEntry with ≥1 additionalEntry | 37,832 |
| additionalEntry elements | 216,068 |
| … with a usable regnum | 215,859 |
| … no regnum (unusable) | 209 |
| … lacking own `<regDate>` (fall back to parent) | 2,189 (1.0%) |
| distinct additionalEntry keys (strict) | 219,524 |
| distinct additionalEntry keys (fallback) | 221,889 |
| strict keys already present as a top-level key | 162 |

## Q3 — Registration-class distribution of the strict net-new joins (informational)

**Class is NOT a scope filter here.** A book MARC can legitimately match a periodical-class
renewal (see `docs/COPYRIGHT_SCENARIOS.md`, the pair-429 finding); this breakdown is for insight
only and excludes nothing. Edges = renewal ↔ additionalEntry-key joins, counted per joining key.

| Class | Edges | Share | Family |
|---|---|---|---|
| A | 8,173 | 67.37% | book |
| BB | 2,408 | 19.85% | periodical (contributions) |
| A5 | 1,175 | 9.69% | periodical contribution (despite 'A') |
| B5 | 216 | 1.78% | periodical contribution |
| AI | 69 | 0.57% | book (ad interim) |
| AF | 51 | 0.42% | book (foreign) |
| AFO | 20 | 0.16% | — |
| AA | 13 | 0.11% | book |
| AIO | 5 | 0.04% | — |
| C | 1 | 0.01% | — |
| **total** | **12,131** | | |
| **book family (A/AA/AF/AI)** | **8,306** | **68.47%** | |

**~68.5% of the recovered joins are directly book-family**, and the periodical-class remainder is
not waste — those renewals can still match in-scope book MARC records.

## Q4 — renewalEntry blocks (minor)

The 153 `<renewalEntry>` elements are **standalone renewal records transcribed directly in the
registration XML** (top-level siblings / inside `<entryGroup>`, NOT `<copyrightEntry>` children —
hence they are invisible to any per-copyrightEntry harvest). Each cites its original registration
under `renewal/registrations/registration` (`<regDate>` + `<regNum>`) and carries its own
`<renewalNum>`.

| | Count |
|---|---|
| renewalEntry elements | 153 |
| citing a registration we hold (would mark it renewed) | 94 |
| renewalNum not already in the renewal TSV corpus (genuinely new) | 137 |

So these would contribute up to **94** additional registration renewals and surface **137** new
renewal records. Tiny in absolute terms, but nearly free and 100% book-context (these volumes are
book renewals). Worth folding into the same #111 work; not a blocker on its own.

## Q5 — Bogus scenario-4 impact

Scenario 4 ("renewal-only, in copyright by an unjoined renewal") queues renewals believed to have
no registration in our corpus. If additionalEntry keys were indexed, some of those candidates would
resolve to a registration and cease to be scenario-4 — i.e. they are **bogus** labeling candidates.

Over `data/renewal_review.db`, all 1,129 `pairing_type='renewal'` rows resolved to their exact
renewal via `(nypl_uuid, cce_renewal_id) == (entry_id, renewal id)` (0 unresolved):

| Outcome | Count | Share |
|---|---|---|
| already joined a top-level key | 82 | 7.26% |
| **become joined via additionalEntry (STRICT) — bogus** | **27** | **2.39%** |
| become joined via additionalEntry (FALLBACK) | 27 | 2.39% |
| still genuinely unjoined | 1,020 | 90.34% |

**27 scenario-4 candidates (2.39%) are bogus specifically because of the dropped additionalEntry
numbers.** Note the separate 82 rows (7.26%) that already join a *top-level* key: those are bogus
for a *different* reason — the builder projects only the first matched renewal per registration
(`_ingest_registrations` breaks after the first hit), so additional renewals sharing that key never
enter the "joined-renewal-id set" the scenario-4 queue filters against. That is its own defect
(worth a follow-up ticket), independent of additionalEntry.

## VERDICT

**Yes — parse `<additionalEntry>` in `nypl_reg.py` (strict own-`<regDate>` keys only) and rebuild
the index.** The evidence is unambiguous:

- **+12,131 real reg↔renewal joins** (+7.25% over the current 167,317), recovering **4.47%** of
  currently-unjoined joinable renewals — a large, first-class gain, not a rounding-error tail.
- **68.5% book-family**, and the periodical-class remainder is still matchable against book MARC.
- **Nearly free and low-risk:** 99% of additionalEntries carry their own `<regDate>`, so the
  trustworthy STRICT key set captures essentially all of the yield. The parent-date fallback adds
  only 58 joins (+0.5%) and risks spurious matches — **do not implement the fallback.**
- **Removes 2.39% of scenario-4 labeling noise** (27 of 1,129 candidates are bogus), improving the
  training queue's signal.

### Recommendation

1. In `nypl_reg.py`, extract each `<additionalEntry>`'s own `regnum` (attribute preferred over
   inline `<regNum>` text) and its own strict `<regDate>`; emit an additional join key per
   additionalEntry via `make_renewal_keys`. **Do not fall back to the parent regDate.**
2. Index those keys so a renewal citing an interior number joins (and flip the parent
   registration's `was_renewed` when any interior key matches). The exact record representation
   (extra keys on the parent vs. additionalEntries as their own retrievable units) is an
   implementation choice for #111; the join-key yield above holds either way.
3. Rebuild the index (the parser fingerprint change invalidates the cache automatically).
4. Follow-ups, separate from #111: the 153 `<renewalEntry>` blocks (94 would-be reg renewals,
   137 novel renewals) and the first-renewal-only projection that produced the 82 already-joined
   scenario-4 rows.
