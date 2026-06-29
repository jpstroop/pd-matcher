# Copyright scenarios

How a MARC↔CCE match maps to copyright-status *determinability*, and why one of the four cases is a public-domain trap.

This document is for library and library-IT readers deciding what a `pd-matcher` result actually tells them. It assumes you know MARC and the basics of U.S. copyright (registration vs. renewal, what the [CCE](GLOSSARY.md) is). Unfamiliar matching or tooling terms are defined in the [glossary](GLOSSARY.md). For the algorithm itself see [DESIGN.md](DESIGN.md) and [MATCHING_ARCHITECTURE.md](MATCHING_ARCHITECTURE.md).

`pd-matcher` is a **linkage producer, not a public-domain determiner**. It surfaces verified links between a library MARC record and the U.S. Copyright Office's registration and renewal records (the CCE, transcribed by NYPL) and the evidence behind them. A human or a downstream consumer then applies the actual copyright reasoning — the 1909-Act notice requirement, the 1964–77 automatic-renewal rule, [URAA](GLOSSARY.md) restoration, country-of-origin analysis — to those links. The four scenarios below describe what kind of determination each match result *enables*, never a determination the tool makes for you.

A legal caution that the scenarios turn on: under the 1909 Act, copyright was secured by publication with notice and had to be *renewed* in its 28th year to stay in force. A matched renewal therefore proves a renewal was filed; it does **not** prove that a clean, separate formal registration existed. The reg↔renewal relationship is something we reconstruct from the data, not a fact the data hands us directly.

---

## The four scenarios

Each scenario is defined *per MARC record*, by what the matcher found (or didn't) on the registration and renewal sides.

### 1. No match — registration absent, renewal absent

No registration and no renewal cleared the score floor for this MARC record. Status is **undetermined**: the work may be genuinely unregistered (and so never had a U.S. term to begin with), or it may be registered under a transcription the matcher couldn't reach. A no-match is an absence of evidence, not evidence of absence — it does not establish public domain on its own.

### 2. Registration matched, renewal joined — registered *and* renewed

A registration matched, and that registration is already linked to a renewal in the CCE data: its `was_renewed` flag is `True` (the precomputed join, below). The work was both registered and renewed, so its initial term was extended. For a consumer, this is the cleanest "still likely under copyright" signal the data offers.

### 3. Registration matched, renewal unlinked — the false-PD risk

A registration matched, but its `was_renewed` flag is `False` — no renewal was joined to it. Read naively, this says "registered, never renewed," which is exactly the 1909-Act pattern that *generally* falls into the public domain. That reading is a **trap**.

`was_renewed=False` is an ambiguous *observable state*, not a verdict: it means only that *the precomputed registration↔renewal join found no renewal* — not that no renewal exists. A renewal can be real and still go unjoined (the renewal side may carry a date or a registration number the join couldn't reconcile, or the renewal may sit in a part of the CCE the registration corpus doesn't cover). So before anyone trusts the "not renewed" reading, the **renewal matcher** re-examines the work — it goes looking for the renewal the join missed. That re-examination resolves the single `was_renewed=False` state into one of two fundamentally different outcomes, distinguished by whether a renewal is a *fact in the data* (3a) or an *inference of its absence* (3b).

#### 3a. Registration matched, renewal exists but unlinked — in copyright

The renewal record *is* in the NYPL data; the join simply missed it (a date or number it couldn't reconcile, or a coverage gap). The renewal matcher finds it. A found renewal is a **positive, verifiable MARC↔renewal link**: the work was renewed, so it is **in copyright**. This is the false-PD trap *caught* — the save. Because it is a real link between two records, it is a pair you can label and study like any other.

#### 3b. Registration matched, no renewal found — the public-domain candidate

The renewal matcher searches and finds nothing. This is the project's headline outcome — **registered, not renewed**, the 1909-Act public-domain pattern — but be precise about what kind of claim it is. There is no renewal record to point at, so 3b is an **inference from absence, not a fact in the data**. We can never *prove* a work was never renewed; the strongest defensible claim is "we searched both the precomputed join *and* the renewal matcher and found nothing." It is therefore inherently **softer** than 3a: a renewal could still exist in some CCE corpus we don't hold. And it is **not a MARC↔renewal pair** — there is no second record to link to. It is a record-level assertion *about the registered work*: "renewal-searched, none found."

The whole reason to separate 3a from 3b is this data-fact-versus-inference distinction. 3a is hard evidence — a link you can verify; 3b is an inference — an absence you can only attest you searched for. Scenario 3 is the ambiguous observable, and 3a and 3b are what it resolves into.

This resolution is a *process* (the renewal matcher plus a re-examination of the verified vault, issue [#107](https://github.com/jpstroop/pd-matcher/issues/107)), and the data model does not yet record its outcome: `was_renewed=False` is a single boolean that cannot distinguish 3b (renewal-searched, confirmed none) from "not yet re-examined," and 3b — having no second record — is not a pair the labeling queue can surface today. The schema and labeling-tool design for recording the renewal-search outcome is tracked as issue [#109](https://github.com/jpstroop/pd-matcher/issues/109).

### 4. Renewal-only — renewal matched, registration not

No registration matched, but a renewal matched the MARC record directly. This is renewal evidence the registration side missed entirely. It tells you a renewal was filed for this work (so a downstream consumer should *not* treat it as unrenewed), even though the matcher couldn't surface the corresponding registration. Per the legal caution above, a renewal here does not retroactively prove a clean registration — it proves a renewal.

Scenarios 3 and 4 are both defined by *registration-match status*, and registration matches from a raw matcher run are candidates, not facts. So both scenarios are built and studied from **verified** registration information — the labeled vault — rather than from unverified matcher output. Every published row is human-verified.

---

## The registration↔renewal join

Scenario 2 versus scenario 3 hinges entirely on one precomputed fact: did a renewal join this registration? Here is how that join is built and why it is trustworthy.

### How the key works

A renewal record cites the *original* registration it renews — its original registration number (`oreg`) and original registration date (`odat`). A registration record carries its own number (`regnum`) and date (`reg_date`). The join pairs them on a composite key: the **normalized registration number plus the registration date**, assembled identically on both sides as `normalize_regnum(regnum)|isoformat(date)`. The code lives in `make_renewal_key` / `make_renewal_keys` in [`src/pd_matcher/index/codec.py`](../src/pd_matcher/index/codec.py); the registration number is canonicalized (regnum normalization is issue [#102](https://github.com/jpstroop/pd-matcher/issues/102)) and multi-number "range" registrations are fanned out into one key per listed number (issue [#103](https://github.com/jpstroop/pd-matcher/issues/103)) so a renewal citing one interior volume still collides. The join runs once, at index-build time, and is frozen into each registration's `was_renewed` flag — so `match` and the labeling tools read a boolean, not a live lookup.

### Why the date is mandatory

Registration numbers are *reused across years*. A whole-corpus analysis (see [docs/findings/nypl_join_analysis_2026-06-27.md](findings/nypl_join_analysis_2026-06-27.md), 2026-06-27, over all 2,168,402 registrations and 443,693 renewals) found that **67.6% of normalized registration numbers appear in more than one year**. A number alone is hopelessly ambiguous, so the date is part of the key by necessity, not convenience.

### Why the join is trustworthy

The same analysis validated the join's core assumption — that a renewal's `(normalized oreg + odat)` points to at most one registration. Of renewals that join any registration, **99.755%** map to exactly one registration at exact-date granularity (**99.39%** at year-level). Many-to-one cases are **under 0.61%**, and they are structurally benign: composite-work splits and true duplicates are the *same underlying work* (picking any one is harmless), while range/volume overlaps are disambiguated by the renewal's own title. The builder's join count (167,317 registrations with at least one renewal) reconciles exactly with the index metadata. A potential year-level join improvement is tracked as issue [#108](https://github.com/jpstroop/pd-matcher/issues/108).

### What unjoined renewals are — and aren't

The analysis also settled what `was_renewed=False` does *not* mean. About 61% of renewals join no registration at all, but this is a **scope mismatch, not a join defect**: the renewal corpus spans every CCE class (music, art, drama), while NYPL's registration transcription is book-focused. Renewals for non-book classes have no book registration to join, by construction. This is exactly why scenario 3 needs re-examination rather than blind trust — an unjoined renewal is invisible to a registration's `was_renewed` flag.

---

## How the pipeline and labeling map onto the scenarios

The two subsystems each touch different scenarios. Keep them distinct: `pd-matcher` is the matching *engine*; `pd-groundtruth` is the *labeling* subsystem that builds and reviews the training set.

| Scenario | Where it comes from |
|---|---|
| 2 — registered and renewed | The precomputed `was_renewed=True` join (index-build time). |
| 3a — registered, renewal exists but unlinked | The **verified vault**, re-examined for renewals the join missed (issue [#107](https://github.com/jpstroop/pd-matcher/issues/107)); a found renewal becomes a verifiable MARC↔renewal link. |
| 3b — registered, no renewal found | The same re-examination when it turns up nothing; recording this "renewal-searched, none found" outcome is future work (issue [#109](https://github.com/jpstroop/pd-matcher/issues/109)). |
| 4 — renewal-only | The **renewal-first queue** (`pd-groundtruth build-renewal-queue`). |

The vault records *which* CCE pathway surfaced each pair in its `match_source` field — `registration`, `renewal`, or `both` (the `both` value is reserved for a future pipeline that links a pair through both pathways and is not produced yet). On the review side the discriminator is the pair's `pairing_type` (`registration` or `renewal`); the labeling code maps `pairing_type="renewal"` to `match_source="renewal"`.

The scenario-4 queue is **renewal-first**: for every in-scope pool MARC it runs the cheap renewal search first, keeps only books whose best renewal clears the floor, then checks for a registration *within the renewal's `odat` year* (not the MARC's publication year). Only when no registration clears the floor in that year is the book emitted as a `pairing_type="renewal"` candidate for labeling. Build it with:

```
pdm run pd-groundtruth build-renewal-queue --pool data/candidates --index caches/cce.lmdb --out data/review.db
```

The registration check defaults to the **learned** combiner (`--reg-scorer learned`, floor `--reg-min-score 90`), while the renewal arm uses the zero-dependency **weighted-mean** combiner (the default engine combiner) at floor `--min-score 60`, because the renewal pathway is untrained. This is the renewal-first builder for issue [#21](https://github.com/jpstroop/pd-matcher/issues/21), in `src/pd_groundtruth/build_renewal_queue.py`.

Independent MARC↔renewal matching as a first-class `pd-matcher` capability — surfacing scenario-4 links at production scale rather than only as a labeling queue — is tracked as issue [#45](https://github.com/jpstroop/pd-matcher/issues/45).
