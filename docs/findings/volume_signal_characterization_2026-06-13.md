# Volume signal characterization (whole/part) — 2026-06-13

Issue #82, Phase 0. Runs the CURRENT `volume.compat` scorer AS-IS over the whole/part vault slice and attributes every non-correct case to a nameable CAUSE and FIELD, so Phase 1 builds only the changes the data justifies. No `src/` code is modified; the vault is read-only.

## Method

**Slice selection.** From `current_entries(data/label_vault.jsonl)`, an entry enters the study slice if EITHER it is tagged `marc_whole_cce_part` / `cce_whole_marc_part`, OR a whole/part surface form (case-insensitive `v.`, `vol`, `pt.`, `part`, `tome`, `tomo`, `bd.`, `band`, `teil`, `t. N`, `heft`, `fasc`, `livre`, `libro`, `complete`, `collected`, `selected`, `ser.`, `aufl`, `course N`) appears in the free-text `note` OR in any resolved record field (MARC title/extent/edition/notes/series; CCE title/desc/notes/new_matter). The union widens beyond the tagged set so keyword-only cases the labeler never tagged are still characterised.

**Resolution path.** Each entry's MARC is resolved via `build_marc_index` over the candidate pool (`data/candidates`) and its CCE via `NyplIndexLookup.get_registration` over `caches/cce.lmdb` — the same proven read-only path as `scripts/learned_scorer_heldout.py`. Entries whose MARC is absent from the pool or whose CCE is absent from the index are counted and skipped.

**Current scorer, as-is.** `score_volume` runs with a real `ScorerContext` from the production `_build_context`, so `ctx.language` is the record's true language. `_classify_marc` / `_classify_cce` are called directly to capture each side's `(cardinality, part_number)`. No `src/` code is modified.

**Outcome definitions.** The gold expectation for a whole/part case: a `no_match` verdict SHOULD get a LOW volume score (the scorer correctly detecting whole/part disagreement; ≤ 25); a `match` verdict SHOULD get a HIGH score (≥ 100).

- **correctly_scored** — the scorer produced a non-skipped score in the expected band.
- **skipped_but_signal_exists** — the scorer skipped (`skipped=True`); whether a recoverable signal exists is decided in cause attribution. A skip is never correct for an adjudicable whole/part case.
- **misclassified** — a non-skipped score in the WRONG band.

**Cause attribution (non-correct cases only), in priority order.**

1. **signal_in_unread_field** — re-running `_detect_part` / `_is_multivolume_whole` / `_is_collected_title` over the fields the scorer never reads (MARC `notes`/`edition`/`series_titles`, CCE `notes`/`new_matter_claimed`) fires on at least one. The firing field and detector are recorded. This directly validates/kills the issue's “~30% in MARC 500 notes” claim against the real resolution path.
2. **roman_numeral_miss** — both sides classify `part` with different numbers, but a raw read-field token on one side is a Roman numeral equal to an Arabic token on the other (`roman_to_arabic(a) == int(b)`).
3. **word_number_miss** — analogous, a number-word vs a digit BEYOND volume.py's built-in one..ten table (`word_to_int` with the record's language).
4. **pattern_gap_candidate** — the scorer skipped, both sides parse to `unknown`, and no unread field recovers a signal: a human-visible designator no current regex matches (eyeball the representative dump).
5. **genuinely_ambiguous** — no recoverable signal anywhere; correctly beyond reach.

## Headline

- **Slice size**: 404 (tagged 49; keyword-only 355; tagged∩keyword overlap 47)
- **Resolved / scored**: 404
- **Unresolved**: 0 missing in pool, 0 missing in index

| outcome | count |
|:---|---:|
| correctly_scored | 44 |
| skipped_but_signal_exists | 339 |
| misclassified | 21 |

## Cause attribution (Phase-1 priority list)

Ranked by count over all NON-correct cases. `correctly_scored` is shown for completeness.

| cause | count |
|:---|---:|
| pattern_gap_candidate | 200 |
| signal_in_unread_field | 86 |
| genuinely_ambiguous | 74 |
| correctly_scored | 44 |

## Signal-in-unread-field, by field

Counts the cases where each unread field would have fired a detector (a single case can fire on multiple fields, so column sums may exceed the 86 `signal_in_unread_field` cases). Directly validates or kills the “notes carry the signal” claim.

| field | side | cases firing |
|:---|:---|---:|
| `marc.notes` | MARC | 65 |
| `marc.edition` | MARC | 0 |
| `marc.series_titles` | MARC | 5 |
| `cce.notes` | CCE | 32 |
| `cce.new_matter_claimed` | CCE | 0 |

## Representative rows

| marc_control_id | nypl_uuid | cause | field | raw_text | score | expected |
|:---|:---|:---|:---|:---|---:|:---|
| 9917914633506421 | 38FC0F79… | signal_in_unread_field | cce.notes | The Library of Christian classics, v.20-21 | 0 | high |
| 9916365903506421 | 7FB5B4CA… | signal_in_unread_field | marc.notes | Issued in 4 v. in mimeographed form, 1939-40, under title: An outline of formal logic | 0 | high |
| 9928909703506421 | 27B2125D… | signal_in_unread_field | marc.notes | Abstracted in Dissertation abstracts, v. 19 (1958) no. 2, p. 385 | 0 | high |
| 9927314393506421 | EDD7B421… | signal_in_unread_field | marc.notes | 1. That fiery particle: 1862-1914.--v. 2. The Little Digger, 1914-1952 | 0 | high |
| 9952625573506421 | F103ACFF… | signal_in_unread_field | cce.notes | (The reference shelf, vol. VII, no. 6) Bibliography: p. [29]-48 | 0 | low |
| 9917782683506421 | 2AD3CF1D… | signal_in_unread_field | marc.notes | pt. 1. The nature and development of diocesan bureaus.- pt. 2. Catholic charities, Dioces… | 0 | high |
| 993308833506421 | 4791FCAE… | signal_in_unread_field | marc.notes | First published in "Themis," vol. II, September 13-27, 1890, under title: California in 1… | 0 | high |
| 9949153473506421 | D0A3CAAE… | signal_in_unread_field | cce.notes | Ergebnisse der Mathematik und ihrer Grenzgebiete, Bd.40 | 0 | high |
| 9949153103506421 | B268856D… | signal_in_unread_field | cce.notes | Ergebnisse der Mathematik und ihrer Grenzgebiete, Bd.48 | 0 | high |
| 9929737033506421 | D427C78D… | signal_in_unread_field | marc.notes | [v. 1.] Reviews and comments.--v. 2. Five film scripts: Noa Noa, The African queen, The n… | 0 | high |
| 99125488304006421 | F9620F7A… | signal_in_unread_field | marc.notes | "Selected letters": p. [323]-352 | 0 | high |
| 9932564843506421 | 88D07C61… | signal_in_unread_field | cce.notes | Title (transliterated); Chumesh l'Talmidim. CONTENTS.—v. 1. Breishis.—v. 2. Shmos-Vayikra… | 0 | low |
| 9921472173506421 | E5B88B97… | signal_in_unread_field | marc.notes | "Translation of the introductory essay by Barrows Mussey. The selections are from The com… | 0 | high |
| 9954069213506421 | F9846C18… | signal_in_unread_field | marc.notes | On spine: Book two. Social studies | 0 | high |
| 9917009853506421 | 4101D6D2… | signal_in_unread_field | marc.notes | Reprint of the 1914 ed., issued as the University of Wisconsin bull. no. 638, History ser… | 0 | high |
| 9932151833506421 | 4D3A1794… | signal_in_unread_field | marc.notes | v. 1. Divine authority. The kingdom of God. Divine authenticity of the Book of Mormon | 0 | low |
| 9927041283506421 | 49FF47C5… | signal_in_unread_field | marc.notes | v. 1. General survey.--v. 2. Eastern-Pacific.--v. 3. Western Pacific (Tonga to the Solomo… | 0 | low |
| 9928868433506421 | AAEC67BC… | signal_in_unread_field | marc.notes | Reprinted from the Louisiana historical quarterly. Vol. 20, no. 4. October, 1937 | 0 | low |
| 993265753506421 | F128BF2E… | signal_in_unread_field | marc.notes | Vol. 16: Comprehensive index | 0 | high |
| 9929738513506421 | 12F2432E… | signal_in_unread_field | marc.series_titles | National Bureau of Economic Research. General series 96. Economic research: retrospect an… | 0 | high |

## Decision — which Phase-1 changes the data justifies

Three candidate Phase-1 changes, ranked by movable-case count. A change attributing fewer than 3 cases is flagged as not worth the false-positive risk it introduces.

| change | movable cases | verdict |
|:---|---:|:---|
| A) widen consulted fields | 86 | pursue |
| B) roman/word part-number normalisation | 0 | skip (false-positive risk) |
| C) language threading (word-number) | 0 | skip (false-positive risk) |

**Recommendation: pursue the changes above the 3-case floor.** 86 cases are movable in total. Per the issue's phasing, Phase 1 implements them in `volume.py` on a `phase-N-volume-designators` branch with structured-wins precedence (notes scanned last under the same guarded regex) and 100% coverage, then Phase 2 retrains the learned model and reruns the held-out eval and `pdm run regression` to confirm the weighted mean improves without a precision regression before shipping.

> Note: per the vault blind-spot finding, the labeled vault is structurally biased toward `match` pairs the matcher already surfaces; movable-case counts here are an UPPER BOUND on top-1 flips, not a prediction. Phase 2's per-MARC diagnostic is the real gate.
