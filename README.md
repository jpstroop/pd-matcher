# pd-matcher

**A pipeline for building a verified MARC ↔ U.S. copyright-record linkage dataset, so we can train an ML matcher to do this at scale.**

---

## Why this exists

To know whether a book published in the United States between 1923 and 1977 is in the public domain, you need to know whether the work was registered with the U.S. Copyright Office and whether that registration was renewed. The Copyright Office published those records in the **Catalog of Copyright Entries (CCE)** — about 2.2M registrations and 444k renewals across that window. The New York Public Library transcribed the CCE volumes into structured XML and TSV; that's the data we use.

Library catalogs use a different format: **MARC** (MAchine-Readable Cataloging). A MARC record describes a holding — a book, a recording, a score — and is what most libraries publish about their collections. Princeton publishes its full catalog as a MARCXML dump.

These two corpora describe the same underlying works but were never designed to link to each other. ISBNs barely existed during the CCE period. LCCNs appear in many MARC records but not the CCE side. Titles, authors, publishers, and years drift between sources (transcription errors, abbreviation conventions, OCR garbles, edition variants). Matching them is fuzzy work.

**The long-term goal**: a learned matcher — a gradient-boosted model (LightGBM) trained on verified pairs — that outperforms hand-tuned scoring weights and can do this matching across many institutions' catalogs, not just Princeton's. Building it requires a labeled training set of confirmed (match / no_match / unsure) pairs across the full score range. **This project produces both: the labeled training set and the learned matcher that consumes it.** That learned combiner is now built and validated — it beats the weighted-mean baseline on held-out pair-level separation (see [docs/LEARNED_MATCHER.md](docs/LEARNED_MATCHER.md)). The weighted-mean combiner remains the zero-dependency default that bootstraps labeling: it surfaces candidates for human verification, and every verdict grows the vault the learned matcher trains on.

## What this project is

A two-CLI pipeline:

- **`pd-matcher`** — proposes candidate `(MARC record, CCE registration, optional CCE renewal)` triples with per-field scores and a confidence. The **default** combiner is a weighted mean over per-field scorers (no extra dependencies; optionally Platt-calibrated). A **LightGBM learned combiner** is also built and selectable with `--scorer learned`: train it with `pd-matcher train-scorer` (needs the optional `ml` extra), and it emits a calibrated match probability directly. It outperforms the weighted mean on held-out separation — see [docs/LEARNED_MATCHER.md](docs/LEARNED_MATCHER.md).
- **`pd-groundtruth`** — turns those proposals into a labeled corpus. It samples candidates across the score range, serves a local labeling UI, and persists every human verdict to `data/label_vault.jsonl` — the vault, source of truth.

The matcher's role is **candidate surfacing**, not direct publishing. Every published row is human-verified. The matcher's mistakes only matter for labeling throughput, not output quality.

The published artifacts live in a separate data repo, [`jpstroop/cce-marc-linkage`](https://github.com/jpstroop/cce-marc-linkage):
- `matches.jsonl` — confirmed linkages only. The curated table for consumers who just want pairs.
- `training.jsonl` — every adjudicated verdict (match / no_match / unsure). The natural training input for a learned matcher.
- `marc.xml` — MARCXML of every MARC referenced by the vault.

This project does not decide public-domain status. Consumers apply whatever copyright reasoning they want — Cornell's decision matrix, the URAA restoration rules, country-of-origin analysis — to the verified linkage.

## Status

Pre-1.0. Single institution today (Princeton MARC against the NYPL-transcribed CCE). Top-1 linkage precision/recall against the labeled vault is high (~99%), but that metric is in-sample and saturated; the operative measure now is pair-level **separation** — whether a score threshold can auto-decide a pair without a human. On a (deliberately hard) held-out sample the learned combiner reaches ROC-AUC ≈ 0.95 against the weighted mean's ≈ 0.94. The labeled vault (~2,000 verified pairs) is the bottleneck on training-set size and triage thresholds. The labeling subsystem is single-user and local; multi-user labeling behind OAuth is tracked at [GitHub #34](https://github.com/jpstroop/pd-matcher/issues/34).

## Install

Requires Python 3.14+ (standard CPython — **not** the free-threaded `t` build) and [PDM](https://pdm-project.org/). [asdf](https://asdf-vm.com/) is recommended for managing the interpreter pin.

```bash
git clone --recurse-submodules <repo-url>
cd public_domain
pdm install
pdm run pre-commit install
```

The CCE data is pulled in via git submodules under `data/nypl-reg/` and `data/nypl-ren/`. If you forgot `--recurse-submodules`, run `git submodule update --init`.

## Where to go next

| If you want to… | Read |
|---|---|
| Run the pipeline (operator or labeler) | [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — mental model, daily flows, troubleshooting |
| Label pairs | [docs/LABELING_WORKFLOW.md](docs/LABELING_WORKFLOW.md) (operational) + [docs/LABELING_GUIDE.md](docs/LABELING_GUIDE.md) (decision rules) |
| Understand the matching algorithm | [docs/DESIGN.md](docs/DESIGN.md) |
| Understand candidate retrieval vs scoring | [docs/MATCHING_ARCHITECTURE.md](docs/MATCHING_ARCHITECTURE.md) |
| Train and use the learned (LightGBM) matcher | [docs/LEARNED_MATCHER.md](docs/LEARNED_MATCHER.md) |
| Ship a code change | [docs/PHASE_WORKFLOW.md](docs/PHASE_WORKFLOW.md) |
| Look up a term | [docs/GLOSSARY.md](docs/GLOSSARY.md) |
| Read past experiments | [docs/studies/](docs/studies/) (committed studies) and [docs/findings/](docs/findings/) (durable diagnostic records) |
| See active work | [GitHub issues](https://github.com/jpstroop/pd-matcher/issues) |

CLI flag reference is in `--help` on each command:

```bash
pdm run pd-matcher --help
pdm run pd-groundtruth --help
```

## References

- **U.S. Copyright Office — Circular 23, "Copyright Office Records"**: <https://www.copyright.gov/circs/circ23.pdf>. Authoritative breakdown of which records exist for which years. Confirms that **December 31, 1977** is the last day of registrations under the 1909 Copyright Act — the boundary the pipeline's coverage reflects.
- **Internet Archive — copyright records collection**: <https://archive.org/details/copyrightrecords>. OCR / scans of the same CCE volumes NYPL transcribed; useful as a human-readable cross-reference.
- **Cornell University Library — "Copyright Term and the Public Domain in the United States"**: <https://guides.library.cornell.edu/copyright/publicdomain>. Reference matrix for downstream consumers applying copyright reasoning to the linkage dataset; this project does not encode it.

## License

The `pd-matcher` source code is licensed under the **GNU Affero General Public License v3.0 or later** (AGPL-3.0-or-later); see [LICENSE](LICENSE). Distributed or network-deployed modifications must be released under the same license.

This license covers the code only, not the bundled data:
- CCE registration and renewal data are pulled in as NYPL-transcribed submodules; NYPL's transcriptions carry their own licenses.
- The underlying Catalog of Copyright Entries is a work of the U.S. Copyright Office and is in the public domain in the United States.
- Any MARC catalog you match against, and the ground-truth pairings you produce, are your own data under your own terms.
