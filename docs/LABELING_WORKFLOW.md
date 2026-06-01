# Labeling workflow

This is the labeler's operational guide — every command you actually run, in the order you run it, with the trigger conditions for each. If you're shipping code changes, you want [phase-workflow.md](phase-workflow.md) instead.

## 1. Label

```bash
pdm run pd-groundtruth review
```

Opens the local review UI at <http://127.0.0.1:8000>. Verdicts auto-save to both `data/review.db` and `data/label_vault.jsonl`. Ctrl-C stops the server; nothing is lost.

For verdict decisions (match vs. no_match vs. unsure, series-vs-volume rules, translation rules) see [LABELING_GUIDE.md](LABELING_GUIDE.md).

## 2. End of labeling session — commit the vault

```bash
git add data/label_vault.jsonl
git commit -m "vault: <N> labels"
```

Just the vault. No other files. Frequency: any time you stop labeling and want the labels durable in git. Once a day at a minimum.

You don't need to push the code repo. The vault is git-tracked locally and gets re-published via the data repo (step 4 below).

## 3. Rebuild the queue — when scoring changed

```bash
pdm run pd-groundtruth build-queue --rebuild
```

Runs the matcher against the full pool, re-scores every pair, writes a fresh `data/review.db`. Vault verdicts pre-apply automatically — you don't lose any labels.

**Trigger:** any time the matcher's scoring code changed since your last rebuild. Concretely, any merge that touched:

- `src/pd_matcher/scorers/`
- `src/pd_matcher/signals/`
- `src/pd_matcher/normalize/`
- `src/pd_matcher/config/defaults/`
- `src/pd_matcher/match/` (including `combiners/` and `pairing_compiler.py`)
- `src/pd_matcher/idf.py` or any IDF table rebuild

Doc-only or test-only changes do not stale the queue.

**Cadence:** when the user (you) wants, but at minimum before the next labeling session after the matcher changed. Batchable — defer until a cluster of merges has all landed; one rebuild covers all of them.

**Runtime:** ~5–10 min depending on pool size.

## 4. Regenerate and publish the dataset — when the vault grew

```bash
pdm run pd-groundtruth dump-vault-marcs
pdm run pd-groundtruth publish-linkage
cd data/published
git add -A
git commit -m "regenerate from vault @ <N> entries"
git push origin main
cd -
```

Writes the three published files (`marc.xml`, `training.jsonl`, `matches.jsonl`) into the in-tree clone of the [cce-marc-linkage](https://github.com/jpstroop/cce-marc-linkage) data repo, then commits and pushes them there.

**Trigger:** the vault grew by enough labels that re-publishing is worth the noise — usually a few hundred new entries, or whenever you specifically want the public dataset current.

**Read-only against the code repo's vault and pool.** Safe to run mid-session if you want.

## 5. Run the LightGBM diagnostic — every few hundred new labels

```bash
pdm run python scripts/learned_scorer_diagnostic.py \
    > docs/findings/learned_scorer_diagnostic_<YYYY-MM-DD>.md
```

Trains a small LightGBM classifier against the current vault and writes a markdown report covering feature importance vs. current weights, per-pair disagreements, and AUC. ~30 sec. Useful as a periodic sanity check on the hand-tuned scoring; gives the next concrete signal about which scorers are under- or over-weighted.

**Trigger:** at meaningful corpus growth milestones — every ~200 new labels is typical. Output goes to `docs/findings/`; you can commit the file separately.

## 6. Rare: rebuild the CCE index

```bash
pdm run pd-matcher index build \
  --reg-dir data/nypl-reg/xml \
  --ren-dir data/nypl-ren/data \
  --out caches/cce.lmdb
```

**Trigger:** the NYPL submodules (`data/nypl-reg/`, `data/nypl-ren/`) updated since the last index build. Rare — these submodules change infrequently.

## 7. Rare: re-acquire MARCs

```bash
pdm run pd-groundtruth acquire --out-dir data/candidates
```

**Trigger:** Princeton's bibdata published a new dump, OR a filter changed in the acquire stage (e.g., the moving wall advanced). After re-acquire you'll want to step 3 (rebuild the queue).

## Where state lives — quick map

| File / dir | What it is | Who writes it | Stales when... |
|---|---|---|---|
| `data/label_vault.jsonl` | Source of truth for verdicts | You (via review UI) | Never — append-and-upsert |
| `data/review.db` | The queue you label against | `build-queue` | Matcher scoring code changes |
| `data/candidates/` | Filtered MARC pool | `acquire` | Princeton publishes new dump |
| `caches/cce.lmdb` | CCE index | `index build` | NYPL submodules update |
| `data/published/` | Published dataset (separate git repo) | `dump-vault-marcs` + `publish-linkage` | Vault grew |

## Where to find more

- Per-verdict decision rules — [LABELING_GUIDE.md](LABELING_GUIDE.md)
- Algorithm internals — [design.md](design.md)
- Matching vs. scoring separation — [matching-architecture.md](matching-architecture.md)
- Term definitions — [glossary.md](glossary.md)
- Code-shipping workflow (not for labelers) — [phase-workflow.md](phase-workflow.md)
