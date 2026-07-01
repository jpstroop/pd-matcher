# Post-rebuild join-maximization verification (2026-07-01)

Read-only, full-corpus measurement confirming the join-maximization work
(`#108` year-based join keys, `#111` `<additionalEntry>` interior keys) landed
after the `caches/cce.lmdb` rebuild, and producing the two decision numbers the
next phase depends on. Proof script:
`scripts/post_rebuild_join_measure.py`.

Everything is built with the current production surface — `iter_nypl_reg_directory`
/ `iter_nypl_ren_directory` and the current `make_renewal_keys(regnum, year:int)`
— so the complete registration keyspace is byte-identical to what the builder
writes in `pd_matcher.index.builder._registration_join_keys`: every
`make_renewal_keys(regnum, reg_year)` plus every
`make_renewal_keys(additional_regnum, additional_year)` harvested from
`<additionalEntry>` children. The retired exact-date key
(`normalize_regnum(regnum)|isoformat(date)`, top-level only) is reconstructed
locally to attribute the gain; the current codec no longer emits it.

## Reconciliation (methodology check)

| quantity | measured | index meta | delta |
| --- | --- | --- | --- |
| registrations parsed | 2,168,402 | 2,168,402 | 0 |
| renewals parsed | 443,693 | 443,693 | 0 |
| reg-centric joins (complete keyspace) | 173,474 | 173,474 | 0 |

The reg-centric join count reproduces the rebuilt index's `renewal_joins` meta
**exactly**, so the local complete keyspace equals the builder's. Every number
below rests on that identity. Joinable renewals (both `oreg` and `odat` present):
443,646 of 443,693 (26 lack `oreg`, 21 lack `odat`). 37,242 registrations carry
at least one `<additionalEntry>`; the additionalEntry key set holds 219,933
distinct keys.

## Q1 — renewal-centric joins now

Counting from the renewal side (how many of the 443,646 joinable renewals hit a
key in the complete registration keyspace):

| join axis | renewals joined |
| --- | --- |
| **joined NOW (complete keyspace)** | **191,156** |
| via top-level year key | 178,741 |
| via `<additionalEntry>` key | 12,643 |
| exact-date baseline (reconstructed) | 172,355 |

The reconstructed exact-date baseline reproduces the prior finding
(`nypl_join_analysis_2026-06-27.md`, 172,355) to the record, and the top-level
year figure (178,741) matches that finding's year-level **prediction** of
≈178,741 exactly. Both corroborate the reconstruction.

**Total gain over the exact-date baseline: +18,801 renewals (+10.91%).**

### Gain breakdown (renewals joined now but not under exact-date)

| mechanism | renewals |
| --- | --- |
| (a) recovered by **year** only | 6,274 |
| (c) recovered by **both** year + additionalEntry | 112 |
| (b) recovered by **additionalEntry** only | 12,415 |
| year-attributable (a + c) | 6,386 |
| additionalEntry-attributable (b) | 12,415 |
| **sum** | **18,801** |

The split is clean because exact-date joins ⊆ top-level-year joins ⊆ complete
joins. Year granularity reclaims 6,386 renewals whose `odat` and the
registration's `reg_date` disagreed by days; `<additionalEntry>` interior
numbers reclaim 12,415 renewals citing a number the top-level parser dropped.

### additionalEntry contribution confirmed

| comparison | renewals |
| --- | --- |
| additionalEntry over exact-date baseline | 12,527 |
| prior finding (`additional_entry_join_yield_2026-07-01.md`) | 12,131 |
| additionalEntry marginal over the year baseline | 12,415 |

additionalEntry **landed** on the renewal-centric axis: it recovers 12,527
renewals the exact-date baseline missed (within +396 of the prior standalone
measurement, the small delta being the interaction with the now-live year keys)
and 12,415 that even the full year keyspace cannot reach. This is not a
year-only illusion — `additional_join_keys` are populated and contributing.

## Q2 — bogus scenario-4 recheck

`data/renewal_review.db`, `pairing_type='renewal'` rows (renewals surfaced as
registration-less orphans for MARC matching):

| quantity | value |
| --- | --- |
| rows | 1,129 |
| resolved to a corpus renewal | 1,129 (100%) |
| **now join the complete keyspace (bogus scenario-4)** | **308 (27.28%)** |
| still genuine orphans | 821 |

This did **not** fall far below the prior 310/109 — it is essentially
unchanged (308 ≈ 310). Breaking the joining renewals down by mechanism
(distinct renewal ids) explains why:

| join mechanism | renewals |
| --- | --- |
| exact-date (old index would have caught) | 80 |
| year-only (new) | 189 |
| additionalEntry-only (new) | 24 |
| both year + additionalEntry (new) | 1 |
| genuinely unjoined orphan | 791 |

80 of these renewals join by **exact date** — the pre-rebuild index already had
those keys, so the queue was built without excluding even exact-date
registration joins; 214 more join only via the new year/additionalEntry keys.
The conclusion is that the join-max is correct and lives in the index (the
reg-centric count reconciles perfectly), but **the renewal review queue in
`data/renewal_review.db` was not built against the rebuilt complete-keyspace
join** — it still lists renewals that have registrations. The queue must be
regenerated against the rebuilt join (drop any renewal whose `oreg`+`odat` hits
the complete keyspace) before it is used, which removes these 308 rows.

## Q3 — #113 harvest (free verified MARC↔renewal positives)

Vault (`data/training/label_vault.jsonl`, read-only): verified
registration-pathway matches are `verdict=="match"` with `match_source` not
`"renewal"` (pre-schema-7 entries carry `match_source is None`, which the
schema-7 migration backfills to `"registration"`).

| quantity | value |
| --- | --- |
| verified reg-pathway matches | 1,090 |
| resolved to a corpus registration | 1,090 (100%) |
| **registration now joined — HARVESTABLE** | **220 (20.18%)** |
| already stamped `was_renewed` at label time | 193 |
| newly joined by the rebuild | 27 |

Each of these 220 is a human-verified MARC↔registration match whose
registration is deterministically joined to a renewal, so it is a **free,
verified MARC↔renewal training positive** requiring no hand-labeling. 27 of
them became joined only because of the rebuild's join-max. For scale, the vault
currently holds **117** hand-labeled renewal-pathway positives — the harvest
alone (220) is nearly double the entire hand-labeled renewal-positive set.

## VERDICT

**Did the join-max land as expected?** On the index and the renewal-centric
axis, yes, unambiguously. The reg-centric join count reconciles with the index
meta to the record (173,474), renewal-centric joins rose +18,801 (+10.91%) over
the exact-date baseline, the year prediction (178,741) hit exactly, and
`<additionalEntry>` contributed its expected ~12.5k. No red flags on the join
itself.

The one place the expected improvement did **not** show up is the bogus
scenario-4 count (Q2): 308/1,129 (27%) of the renewal review queue still join
the complete keyspace — 80 of them even by exact date. That is a stale-queue
problem, not a join problem: `data/renewal_review.db` predates / does not
consume the rebuilt join and must be regenerated against the complete keyspace.
Until it is, the renewal-orphan pool is ~27% contaminated.

**Is the 2a harvest big enough to lean on instead of hand-labeling scenario-4?**
Yes. 220 free verified MARC↔renewal positives — larger than the entire
117-entry hand-labeled renewal-positive set — is worth harvesting directly
rather than hand-labeling equivalent positives. Caveats: the harvest yields
**positives only** (a renewal matcher still needs negatives), and the "2b
residual" analysis (which registrations/renewals the matcher fails to recover
despite the join existing) is deferred — it needs the renewal matcher and is
out of scope here.
