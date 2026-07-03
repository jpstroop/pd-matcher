# Copyright scenarios

How a MARC↔CCE match maps to a copyright-status *determination* — in copyright, public-domain candidate, or undetermined — and why the lone public-domain signal is an inference you can never prove, not a fact in the data.

This document is for library and library-IT readers deciding what a `pd-matcher` result actually tells them. It assumes you know MARC and the basics of U.S. copyright (registration vs. renewal, what the [CCE](GLOSSARY.md) is). Unfamiliar matching or tooling terms are defined in the [glossary](GLOSSARY.md). For the algorithm itself see [DESIGN.md](DESIGN.md) and [MATCHING_ARCHITECTURE.md](MATCHING_ARCHITECTURE.md).

`pd-matcher` is a **linkage producer, not a public-domain determiner**. It surfaces verified links between a library MARC record and the U.S. Copyright Office's registration and renewal records (the CCE, transcribed by NYPL) and the evidence behind them. A human or a downstream consumer then applies the actual copyright reasoning — the 1909-Act notice requirement, the 1964–77 automatic-renewal rule, [URAA](GLOSSARY.md) restoration, country-of-origin analysis — to those links. The three scenarios below describe what kind of determination each match result *enables*, never a determination the tool makes for you.

A legal caution that the scenarios turn on: under the 1909 Act, copyright was secured by publication with notice and had to be *renewed* in its 28th year to stay in force. Renewal facts here come from one place — the deterministic **registration↔renewal join** (described below), which reconciles a renewal to the registration it cites. That reconstructed link is something we build from the data, not a fact the data hands us directly, and a joined renewal proves a renewal was *filed* — it is not a live copyright status.

---

## The three scenarios

Each scenario is defined *per MARC record*, by whether a registration matched and, if so, whether a renewal is on record for it. The tool reports those facts; it never asserts a status. But the organizing axis is the **determination each result enables**, not the observable state the matcher happens to record. Three determinations cover every case, and the tool makes none of them for you — it surfaces the fact that *enables* one: an **in-copyright** determination (scenario 2, which puts a renewal on record), a **public-domain candidate** (scenario 3), and **undetermined** (scenario 1).

The spine of this document is the difference between scenarios 2 and 3: **scenario 2 is a data fact you trust; scenario 3 is an inference from absence you can never prove.** A joined renewal is a hard fact in the CCE data with no score to weigh; "no renewal on record" is the *absence* of that fact, and absence is never proof. Keep that distinction in view — it is what separates an in-copyright determination from a public-domain candidate.

A renewal is a **dated fact, not a status**: the tool reports that a work was *renewed in year X*, not that it is *currently* in copyright. Renewed works still age out — a 1930 registration renewed in 1958 is entering the public domain now as its ~95-year term expires — so the current-status call (moving wall / term math, the 1909-Act notice requirement, [URAA](GLOSSARY.md) restoration) is always the consumer's.

### 1. No match — undetermined

No registration cleared the score floor for this MARC record. Status is **undetermined**: the work may be genuinely unregistered (and so never had a U.S. term to begin with), or it may be registered under a transcription the matcher couldn't reach. A no-match is an absence of evidence, not evidence of absence — it does not establish public domain on its own.

### 2. Registered + renewal on record — in copyright

A registration matched, and a renewal is on record for it: the matched registration's `was_renewed` flag is `True`. That is the fact the tool reports, and it is exactly what *enables* an in-copyright determination (once the consumer confirms the term hasn't since expired).

This is a **hard data fact, not a matcher result — there is no score to weigh.** The renewal cites this registration by number and year, and the precomputed registration↔renewal join (below) reconciled the two once, at index-build time. Your confidence is the *join's* reliability — not a fuzzy-match probability. You trust it.

### 3. Registered, no renewal on record — public-domain candidate

A registration matched and the precomputed join shows no renewal for it: `was_renewed` is `False`. This is the project's headline outcome — **registered, not renewed**, the 1909-Act public-domain pattern, and the *only* one of the three scenarios that points toward the public domain — but be precise about what kind of claim it is.

There is no renewal record to point at, so scenario 3 is an **inference from absence, not a fact in the data.** We can never *prove* a work was never renewed. The strongest defensible claim is narrow: *we checked the precomputed join over the corpus we hold and found no renewal citing this registration.* That is softer than scenario 2 in two distinct ways. First, the searched space is bounded — a renewal could exist in a part of the CCE we don't hold, and a renewal that never cited its original registration by number and year would not join even if we did hold it. Second, it is the *join*, not any content match, that vouches for the absence; there is no second record to inspect. Scenario 3 is therefore not a MARC↔renewal pair. It is a record-level assertion *about the registered work*: "join-checked, no renewal found."

### The two trust models behind these facts

Every scenario but the no-match (1) rests on at least one matcher result — a registration match — and a raw matcher result is a candidate, not a fact (even in scenario 2, where the trusted join sits on top of a registration match that still had to be made). This is why the **training vault** is built only from **verified** information: every vault row is human-reviewed, so the model learns from facts, not from the matcher's own guesses.

The **produced dataset is a different artifact with a different trust model.** At scale, every in-scope book carries the matcher's *best guess* at its scenario — auto-assigned, not individually human-verified. Its trustworthiness comes from the matcher's *measured precision* (calibrated on the human-verified vault) plus the consumer's own copyright reasoning — `pd-matcher` is a linkage producer, not a determiner. The human-verified vault is the training subset that makes those guesses good; it is not a promise that every published row was inspected by hand. In particular, a scenario-3 (public-domain candidate) row in the output is the matcher's best guess that a registered work has no renewal — a lead for a consumer to confirm, not an adjudication. Scale verification of the produced output (calibrated auto-accept, sampling, cross-matcher agreement) is tracked as issue [#97](https://github.com/jpstroop/pd-matcher/issues/97).

---

## The registration↔renewal join

Scenario 2 versus scenario 3 hinges entirely on one precomputed fact: did a renewal join this registration? Here is how that join is built and why it is trustworthy.

### How the key works

A renewal record cites the *original* registration it renews — its original registration number (`oreg`) and original registration year (from `odat`). A registration record carries its own number (`regnum`) and year (`reg_year`). The join pairs them on a composite key: the **normalized registration number plus the four-digit registration year**, assembled identically on both sides as `f"{normalize_regnum(regnum)}|{year}"`. The code lives in `make_renewal_key` / `make_renewal_keys` in [`src/pd_matcher/index/codec.py`](../src/pd_matcher/index/codec.py).

Three refinements widen the join without loosening it:

- The registration number is canonicalized before assembly (regnum normalization, issue [#102](https://github.com/jpstroop/pd-matcher/issues/102)) so transcription variance — interior spaces, hyphens, verbose foreign/interim class phrases — cannot split an otherwise-valid join. The same normalizer runs on the renewal `oreg` writer and the registration `regnum` reader, so both sides land on the identical key.
- Multi-number "range" registrations (`"A692774 A692775"`) are fanned out into one key per listed number (issue [#103](https://github.com/jpstroop/pd-matcher/issues/103)) so a renewal citing one interior volume still collides.
- Every `<additionalEntry>` interior number a registration carries contributes its own key too (issue [#111](https://github.com/jpstroop/pd-matcher/issues/111)), so a renewal citing an interior number marks the parent registration `was_renewed` even when the top-level number never joins. The builder walks the top-level key first, then each additionalEntry key, in `_registration_join_keys` in [`src/pd_matcher/index/builder.py`](../src/pd_matcher/index/builder.py).

The join runs once, at index-build time, and is frozen into each registration's `was_renewed` flag — so `match` and the labeling tools read a boolean, not a live lookup.

### Why the year, not the exact date

The join keys on the registration *year*, not the full date — a deliberate widening that shipped as issue [#108](https://github.com/jpstroop/pd-matcher/issues/108). Two facts force it. First, a number alone is hopelessly ambiguous: about two-thirds of normalized registration numbers are reused across years, so *some* date component is required to disambiguate. Second, exact-date agreement is too strict — a renewal's `odat` and its registration's `reg_date` routinely differ by a few days while naming the same registration, and tens of thousands of registrations carry a derived year with no transcribed `<regDate>` at all. Keying on the year joins both cases; the earlier exact-date key silently dropped them. The registration side supplies `reg_year` (its `regDate → copyDate → pubDate` fallback) and the renewal side supplies `odat.year`, and the two align because `reg_year` equals `reg_date.year` whenever a `<regDate>` exists.

### Why the join is trustworthy

The join's core assumption is that a renewal's `(normalized oreg + year)` points to at most one registration. A whole-corpus analysis validated it: of renewals that join any registration, **99.39%** map to exactly one registration at year-level granularity (see [docs/findings/nypl_join_analysis_2026-06-27.md](findings/nypl_join_analysis_2026-06-27.md), 2026-06-27, over all 2,168,402 registrations and 443,693 renewals). The remaining ~0.61% many-to-one cases are structurally benign: composite-work splits and true duplicates are the *same underlying work* (picking any one is harmless), while range/volume overlaps are disambiguated by the renewal's own title.

The current join counts come from the post-rebuild verification (see [docs/findings/post_rebuild_join_2026-07-01.md](findings/post_rebuild_join_2026-07-01.md), 2026-07-01): **173,474 registrations** carry at least one renewal (the reg-centric count, which reconciles to the index's `renewal_joins` metadata exactly), and **191,156 renewals** join a registration we hold (the renewal-centric count). The year-level and `<additionalEntry>` keys together recovered +18,801 renewals (+10.91%) over the retired exact-date key — 6,386 from year granularity and 12,415 from interior additionalEntry numbers. To reproduce the numbers, point the proof script named in that findings doc at a fresh index build.

### What unjoined renewals are — and aren't

The bulk of renewals join no registration in our corpus, and it is important to read that correctly: it is a **scope mismatch, not a join defect.** The renewal corpus spans every CCE class — music, art, drama — while NYPL's registration transcription (and this project's scope) is book-focused. Renewals for non-book classes have no book registration to join, by construction. An unjoined renewal is therefore invisible to any registration's `was_renewed` flag simply because the matching registration isn't in the corpus we hold.

This is also why `pd-matcher` no longer matches MARC records to renewals directly. A 2026-07-03 census of the renewal index settled the question: of 427,510 renewals, only 26 lack an original-registration number (`oreg`) and 20 more lack its year — **99.989% cite their original registration by number and year.** That citation *is* the deterministic link. Fuzzy content-matching a MARC record to a renewal adds nothing over following the citation the renewal already carries, so the renewal-matching subsystem was removed (2026-07-03; the [#45](https://github.com/jpstroop/pd-matcher/issues/45) ticket keeps the full record and remains the place to revisit the decision). For any unjoined renewal, the practical consequence is the same: its cited `oreg` number and year are themselves the lead a consumer can follow to the original registration, wherever that registration lives.

---

## How the pipeline and labeling map onto the scenarios

Two subsystems touch these scenarios; keep them distinct. `pd-matcher` is the matching *engine* — it produces MARC↔registration pairs and stamps each matched registration's `was_renewed` fact from the precomputed join. `pd-groundtruth` is the *labeling* subsystem that builds and reviews the human-verified training set.

| Scenario | Where the fact comes from |
|---|---|
| 2 — registered + renewal on record | The matched registration's `was_renewed=True` flag (the precomputed join, frozen at index-build time). |
| 3 — registered, no renewal on record | The same matched registration's `was_renewed=False` flag — the join found no renewal citing it. |

The vault records *which* CCE pathway surfaced each pair in its `match_source` field. Every pair produced today is `registration` (a MARC↔registration match); the historical `renewal`-pathway pairs from the removed renewal arm survive only in the archived `data/training/renewal_arm_pairs.jsonl` and are not regenerated.
