# Recall-miss forensics — 2026-07-03

A case-by-case autopsy of every registration-arm recall miss against the current vault, to separate genuine scorer failures from misses that are only misses because the "gold" registration and the one the matcher picked are two registrations of the *same work*. The forensic pass turned up a class of near-invisible false misses — the matcher chose a valid sibling registration — and drove three same-day shipments (the class-token fold, sibling renewal-fact propagation, and the copyright-facts output columns).

Pair numbers below are this eval run's numbering and reset on every rebuild; the durable identifiers are the CCE registration numbers, which are stable across rebuilds.

## The misses

57 registration-arm misses total, split two ways:

- **49 no-top-1** — no candidate cleared the floor, or the true registration was out-ranked below rank 1 by a genuinely different registration. These are ordinary scoring failures.
- **8 wrong-top** — a *different* registration out-ranked the gold one at rank 1.

Of the 8 wrong-tops, **6 are false misses**: the matcher's top-1 is a valid registration of the same work, just not the one the vault labeled. Only 1 is a genuine scorer error; 1 more is a defensible sparse-record artifact.

### Wrong-tops taxonomy

| pair | registrations involved | why it's a (false) miss |
|---|---|---|
| 2 | `A193774` (prev-regNum `AI-4671`) | ad-interim + full registration of one work |
| 4 | ad-interim + full pair | same as pair 2 |
| 10 | ad-interim + full pair | same as pair 2 |
| 8 | `A933140` "© on text" vs `A933141` "© on illus." | component-claim twins |
| 44 | `A789058` / `A789059` | component-claim twins |
| 3 | sparse same-title duplicate | missing fields dodge penalties, out-scores the rich record |
| 15 | series-level registration vs the book | different bibliographic unit |
| 7 | — | the one genuine scorer error |

**Ad-interim + full registration (pairs 2/4/10).** A work registered *ad interim* (a book-class foreign registration, class `AI`) and then fully registered carries two registration numbers for one work, cross-linked by `<prev-regNum>`. Pair 2 is the canonical example: the full registration `A193774` back-references the ad-interim `AI-4671`, and the single renewal `RE188301` carries **two** original-registration rows — `A193774` and `AIO-4671`. Whichever number the vault labeled, the other is an equally-correct top-1.

**Component-claim twins (pairs 8/44).** One physical book gets two adjacent registrations for different copyrightable components — `A933140` claims copyright "on text", `A933141` "on illus[trations]"; likewise `A789058`/`A789059`. Both describe the same book and score near-identically; picking the sibling is not an error.

**Sparse same-title duplicate (pair 3).** A thin registration with the same title as the gold one out-scores it because its *missing* fields dodge the scorers that would have penalized a mismatch — an absent publisher or author cannot disagree. Defensible behavior, but a reminder that absence is not agreement.

**Series-level registration (pair 15).** The matcher picked a registration for the series rather than the individual book — a genuine whole/part confusion, the province of the volume/part signals (#82), not a false miss.

**Genuine error (pair 7).** One, and only one, of the eight is an ordinary wrong pick with no same-work excuse.

### The 49 no-top-1 misses

These are ordinary scoring failures — the true registration is in the candidate pool but out-ranked or below the floor (the live #20 failure mode). Their class profile is 42 domestic book (`A`), 6 foreign book (`AF`), 1 ad-interim book (`AI`), which is proportional to the vault's class mix — nothing class-specific is being systematically dropped. They feed the ordinary scorer-fix ladder (IDF-weighted names #83, whole/part #82, OCR title noise #55), not a structural fix.

## The renewal-citation census (evidence for #45)

Renewals almost always cite the registration they renew, which is why the join backbone — not a separate renewal matcher — is the right tool. Of **427,510** renewals in the index, **26** lack an `oreg` (original registration number) and **20** more lack the renewal date, so **99.989%** cite their original registration. This census is the evidence recorded on **#45** for removing renewal-only matching in favor of the join.

## The byte-level surface-variance find

Tracing pair 2's `AIO-4671` exposed a transcription inconsistency the join key never accounted for. The ad-interim/foreign class marker is transcribed inconsistently on the two sides, and the difference is literally at the byte level:

- Renewal citation `AIO-4671` — the character after `AI` is the **letter O** (`0x4f`).
- Registration `AF0-76081` — the character after `AF` is the **digit zero** (`0x30`).

Both are class noise (`AI` = book ad-interim, `AF` = book foreign), not part of the serial number, but a byte-for-byte join key treats `AIO4671`, `AI04671`, and `AI4671` as three different registrations, so the renewal silently fails to join. `normalize_regnum` now folds the stray `O`/`0` after an `AI`/`AF` class (see [MATCHING_ARCHITECTURE.md](../MATCHING_ARCHITECTURE.md#registration-number-normalization-and-the-class-token-fold)).

## Consequences shipped the same day

1. **Class-token fold** in `normalize_regnum` — recovered **+373** renewal joins on the current corpus (173,474 → 173,847). (An earlier scan estimated only 28; it modeled just the top-level regnum keys and missed the interior `<additionalEntry>` joins, so the shipped, full-keyspace delta is an order of magnitude larger.)
2. **Sibling renewal-fact propagation** at index build — `<prev-regNum>`-linked registration groups now share renewal facts, so the ad-interim record of a renewed pair reports the renewal even though its own number never joined. **223** records were rewritten on the current corpus.
3. **Copyright-facts output columns** — `match_regnum`, `match_prev_regnums`, `match_was_renewed`, `match_renewal_id`/`_date`/`_via` now surface the registration and renewal facts (including the prior-registration back-references) in the `match` JSONL, so a consumer can see the whole-work picture the forensic pass had to reconstruct by hand (see the [output-columns reference](../USER_GUIDE.md#whats-in-each-output-row)).
