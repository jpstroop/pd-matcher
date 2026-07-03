# Best-of-element publisher/claimant scoring — 2026-07-03

Measurement and result for the `combine: best` pairing mode (issue #86). A CCE registration routinely carries several names in one list — a publisher *and* one or more person co-claimants ("Putnam" plus "James D. Horan"). The prior `concat` combine joined that list into a single blob before scoring it against the MARC publisher, so a correct publisher match ("Putnam" vs "Putnam") was diluted by the co-claimant tokens that share nothing with the MARC publisher. `best` keeps each list element separate, scores the MARC field against each, and keeps the single best Evidence for the group. This documents the A/B that justified shipping it as the default.

Frozen proof: `scripts/publisher_split_ab.py`. Read-only: it resolves the labeled vault pairs against the index and scores the publisher group both ways (joined blob vs best-of-element); it never writes the vault or the index.

## Corpus

- **Labeled pairs scored**: 1,934
- **Pairs with a multi-name CCE publisher list**: 814 (42%)

Only the 814 multi-name pairs can move — a single-name list scores identically under both modes.

## Result

Best-of-element scoring versus the joined blob, over the 1,934 pairs:

| measure | joined | best-of-element |
|---|---:|---:|
| publisher-evidence AUC | 0.8187 | **0.8293** |
| pairs lifted ≥ 0.05 (positives) | — | 244 |
| pairs lifted ≥ 0.05 (negatives) | — | 41 |

The lift is asymmetric in the right direction: it raises 244 true-match publisher scores against only 41 no-match ones, so the publisher signal separates positives from negatives better (AUC +0.0106). The negatives that lift are co-claimant lists whose *person* element happens to fuzzy-match the MARC publisher — the residual #86 concern (see below).

Representative lifts (`marc_publisher | cce_names | joined → best`):

- `Putnam` | `['Putnam', 'James D. Horan']` | 0.36 → 1.00
- `University of Arizona Press` | `['University of Arizona Press', 'Board of Regents of the Universities & State College of Arizona']` | 0.53 → 1.00
- `Moody Press` | `['Moody Press.', 'Moody Bible Institute of Chicago']` | 0.42 → 1.00
- `Harper` | `['Harper', 'Frederick Benjamin Gipson']` | 0.55 → 1.00
- `Faber and Faber` | `['Faber and Faber', 'Thomas Stearns Eliot']` | 0.47 → 1.00

## Shipped

Landed as `combine: best` on the `publisher_names` and `claimants` CCE fields in `field_pairings.yaml` (the mechanism is described in [DESIGN.md](../DESIGN.md#step-3-configurable-field-pairings)). Regression against the live vault after the change:

| metric | before | after |
|---|---:|---:|
| recall | 0.9477 | **0.9558** |
| precision | — | 0.9905 |
| F1 | — | 0.9733 |

Recall rose +0.0081 (nine more registration-arm records get the correct top-1). The refreshed baseline is P **0.9905** / R **0.9567** / F1 **0.9733**.

## What this does *not* close

Issue #86 stays open. `best` fixes the *dilution* half of the problem — a real publisher match is no longer averaged down by co-claimant names. It does **not** add person-vs-organization gating: a person co-claimant whose name fuzzy-matches the MARC publisher can still lift a no-match pair (the 41 negatives above). Distinguishing "this element is an organization, score it as a publisher" from "this element is a person, don't" is the remaining #86 lever.
