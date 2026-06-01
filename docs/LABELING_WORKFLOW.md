# Labeling workflow

The labeler's operational guide. If you're shipping code changes, see [phase-workflow.md](phase-workflow.md) instead.

The shape:

- **Routine** — two steps you do every session.
- **Discretionary** — tools you reach for whenever they're useful.
- **Required when** — rebuilds that fire only on a specific condition.

---

## Routine — every session

### 1. Label

```bash
pdm run pd-groundtruth review
```

Opens the local review UI at <http://127.0.0.1:8000>. Verdicts auto-save to both `data/review.db` and `data/label_vault.jsonl`. Ctrl-C stops the server; nothing is lost. For verdict decisions, see [LABELING_GUIDE.md](LABELING_GUIDE.md).

### 2. Commit the vault

```bash
git add data/label_vault.jsonl
git commit -m "vault: <N> labels"
```

Just the vault — no other files. Whenever you stop labeling; once a day at a minimum.

---

## Discretionary — run whenever useful

### Run the LightGBM diagnostic

```bash
pdm run python scripts/learned_scorer_diagnostic.py \
    > docs/findings/learned_scorer_diagnostic_<YYYY-MM-DD>.md
```

Trains a small LightGBM classifier against your current vault and writes a markdown report: feature importance vs current weights, per-pair disagreements, AUC. ~30 sec.

Useful for catching your own labeling mistakes and surfacing scoring/feature nuances. Run as often as you want — the output is a dated snapshot.

### Regenerate + publish the dataset

```bash
pdm run pd-groundtruth dump-vault-marcs
pdm run pd-groundtruth publish-linkage
cd data/published
git add -A
git commit -m "regenerate from vault @ <N> entries"
git push origin main
cd -
```

Writes `marc.xml`, `training.jsonl`, and `matches.jsonl` into the [cce-marc-linkage](https://github.com/jpstroop/cce-marc-linkage) data repo, commits, and pushes. Run whenever you want the public artifact current.

Read-only against the code repo's vault and pool. Safe to run mid-session.

---

## Required when — only fires on a condition

### Rebuild the queue

```bash
pdm run pd-groundtruth build-queue --rebuild
```

**When:** the matcher's scoring code changed since your last rebuild. Specifically, any merge touching:

- `src/pd_matcher/match/` — scorers, signals, combiners, pipeline, IDF, pairing compiler.
- `src/pd_matcher/normalize/` — tokenization, stemming, stopwords, numbers, script detection.
- `src/pd_matcher/config/defaults/` — `field_pairings.yaml` and other matcher configs.

**Skip when:** doc-only, test-only, or unrelated code changes. Without a scoring change, the queue isn't stale.

Vault verdicts pre-apply automatically — no labels are lost. Runtime: ~5–10 min.

### Rebuild the CCE index

```bash
pdm run pd-matcher index build \
  --reg-dir data/nypl-reg/xml \
  --ren-dir data/nypl-ren/data \
  --out caches/cce.lmdb
```

**When:** the NYPL submodules (`data/nypl-reg/`, `data/nypl-ren/`) updated, OR parser code (`src/pd_matcher/parsers/nypl_reg.py`, `parsers/nypl_ren.py`) changed.

Rare. After rebuilding the index, also rebuild the queue.

### Re-acquire MARCs

```bash
pdm run pd-groundtruth acquire --out-dir data/candidates
```

**When:** Princeton's bibdata published a new dump, or the acquire filter changed (e.g., the moving wall advanced).

Rare. After re-acquire, rebuild the queue.

---

## The cycle

The diagnostic surfaces things worth shipping — under-weighted scorers, missing signals, false-positive clusters. Acting on those is a code-shipping job (ask whoever is in the developer seat), not a labeler's job. When the change lands, it usually means a queue rebuild before your next session.

```
label → commit vault → [diagnostic] → ask for code changes → [rebuild queue]
   ↑                                                              │
   └──────────────────────────────────────────────────────────────┘
```

---

## Where state lives

| File / dir | What it is | Who writes it | Stales when... |
|---|---|---|---|
| `data/label_vault.jsonl` | Source of truth for verdicts | You (via review UI) | Never — append-and-upsert |
| `data/review.db` | The queue you label against | `build-queue` | Matcher scoring code changes |
| `data/candidates/` | Filtered MARC pool | `acquire` | Princeton publishes new dump |
| `caches/cce.lmdb` | CCE index | `index build` | Parser code or NYPL submodules change |
| `data/published/` | Published dataset (separate git repo) | `dump-vault-marcs` + `publish-linkage` | Vault grew |

## Where to find more

- Per-verdict decision rules — [LABELING_GUIDE.md](LABELING_GUIDE.md)
- Algorithm internals — [design.md](design.md)
- Matching vs scoring separation — [matching-architecture.md](matching-architecture.md)
- Term definitions — [glossary.md](glossary.md)
- Code-shipping workflow (not for labelers) — [phase-workflow.md](phase-workflow.md)
