# Phase workflow

How a single improvement to the matcher ships, from clean main to merged main, with all gates and the apples-to-apples per-pair diff that lets us spot regressions the aggregate metrics hide.

This is the project's contributor workflow. For codebase architecture see [design.md](design.md); for matching internals see [matching-architecture.md](matching-architecture.md).

## Why this shape

The matcher's behavior is hard to observe through aggregate metrics alone. Precision and recall on the regression baseline can move by a few thousandths and conceal: (a) a handful of true wins balanced by a handful of new false positives, (b) score moves that strengthen correct predictions without crossing a threshold, or (c) shifts driven by corpus growth rather than the change itself. A per-pair diff against `main` resolves all three by surfacing actual flip/no-flip behavior on the same vault.

So every change ships as an isolated phase branch with its own regenerated baseline, and merge-readiness is judged by a per-pair diff against `main` — not by the aggregate delta alone.

## The flowchart

```
┌─────────────────────────────────────────────────────────────┐
│ START on main, clean tree                                   │
│   git status --short && git log --oneline -1                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. CREATE PHASE BRANCH                                      │
│   git checkout -b phase-N-topic                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. IMPLEMENT (code, tests, then gates)                      │
│   <edit files>                                              │
│   pdm run pytest                       # tests + 100% cov   │
│   pdm run mypy src tests               # types              │
│   pdm run ruff check src tests         # lint               │
└─────────────────────┬───────────────────────────────────────┘
                      │ all gates green
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. REGENERATE BASELINE (~10 min, full vault re-scoring)     │
│   pdm run regression-baseline                               │
│   → writes tests/regression/baseline.json                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. COMMIT (no --amend; new commit if pre-commit hook fails) │
│   git add <files> tests/regression/baseline.json            │
│   git commit -m "topic: short summary (closes #N)"          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. PER-PAIR DIFF — apples-to-apples score comparison        │
│   pdm run python scripts/diff_branch_predictions.py \       │
│       dump /tmp/scores_branch.jsonl                         │
│   git checkout main                                         │
│   pdm run python scripts/diff_branch_predictions.py \       │
│       dump /tmp/scores_main.jsonl                           │
│   pdm run python scripts/diff_branch_predictions.py \       │
│       diff /tmp/scores_main.jsonl /tmp/scores_branch.jsonl  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
              ┌───────┴────────┐
              │ READ THE DIFF  │
              └───────┬────────┘
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
  ┌─────────┐  ┌────────────┐  ┌──────────────┐
  │ NO-MATCH│  │ MATCH      │  │ Score moves  │
  │ → MATCH │  │ → NO-MATCH │  │ no flip      │
  │ flips   │  │ flips      │  │ (≥ 0.10)     │
  ├─────────┤  ├────────────┤  ├──────────────┤
  │vault=   │  │vault=      │  │direction     │
  │match:   │  │match:      │  │matters       │
  │TRUE WINS│  │REGRESSIONS │  │  matches up  │
  │         │  │(should be  │  │  = good      │
  │vault≠   │  │ ~0)        │  │  no_match    │
  │match:   │  │            │  │  down = good │
  │POTENTIAL│  │vault≠match:│  │              │
  │FALSE POS│  │true negs   │  │              │
  │(verify) │  │newly fixed │  │              │
  └─────────┘  └────────────┘  └──────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. SPOT-CHECK ANY SUSPICIOUS PAIRS                          │
│   sqlite3 data/review.db \                                  │
│     "SELECT id FROM review_pair WHERE marc_control_id='X'   │
│      AND nypl_uuid='Y';"                                    │
│   → open http://127.0.0.1:8000/pair/<id>                    │
│   (pdm run pd-groundtruth review if the server is not up)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
              ┌───────┴────────┐
              │ DECIDE: MERGE? │
              └───────┬────────┘
                      │ yes
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. MERGE --no-ff                                            │
│   git checkout main                                         │
│   git merge --no-ff phase-N-topic \                         │
│     -m "Merge phase-N-topic: summary (#N)"                  │
│                                                             │
│   ↳ If conflict on baseline.json (typical when two branches │
│     touched it):                                            │
│       git checkout --theirs tests/regression/baseline.json  │
│       pdm run regression-baseline                           │
│       git add tests/regression/baseline.json                │
│       git commit                                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. POST-MERGE GATE CHECK                                    │
│   pdm run pytest && pdm run mypy src tests \                │
│     && pdm run ruff check src tests                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. CLEAN UP                                                 │
│   git branch -d phase-N-topic                               │
│   gh issue close N -R jpstroop/pd-matcher -c "merged at …"  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 10. REBUILD QUEUE (if scoring changed; batchable)           │
│    pdm run pd-groundtruth build-queue --rebuild             │
│    (defer until a cluster of related branches lands)        │
└─────────────────────────────────────────────────────────────┘
```

## Reading the diff

`scripts/diff_branch_predictions.py` (the script invoked in step 5 of the flowchart above) reports three categories per pair (default threshold 0.5, default move threshold 0.10):

1. **NO-MATCH → MATCH flips.** Score crossed up through the threshold. Split by vault verdict:
   - `vault=match` rows are **true wins** — the branch promoted a labeled match that `main` had below the line.
   - `vault!=match` rows are **potential false positives** worth spot-checking before merge. A labeling mistake on our side is also possible; spot-check resolves which.
2. **MATCH → NO-MATCH flips.** Score crossed down. These are **regressions** when `vault=match` and should be near zero. `vault!=match` rows in this column are **true negatives newly correct** — the branch demoted a false positive.
3. **Large moves without flipping.** Score shifted by `≥ move_threshold` but stayed on the same side of the threshold. Direction matters: labeled matches moving up and labeled no-matches moving down are both wins; the reverse is a soft regression.

## Invariants

- **Never amend.** New commits only; a failing pre-commit hook means re-stage and commit again, not `--amend`.
- **Never touch `data/label_vault.jsonl` during branch ops.** The vault is append-only and labeled by hand; commit it separately on `main` only. Run `git stash list` after any agent delegation to verify nothing stashed it.
- **Two branches both moved `baseline.json` → regenerate post-merge on main.** Neither branch's baseline is the combined-effect baseline; trusting one of them publishes a misleading number.
- **Vault parity for cross-branch diffs.** When the labeled corpus has grown on `main` since the branch was created, copy `main`'s vault into the branch's working tree before dumping (`git show main:data/label_vault.jsonl > data/label_vault.jsonl`), dump, then restore (`git checkout HEAD -- data/label_vault.jsonl`). The diff script requires identical vault rows on both sides to be meaningful.

## When to skip the diff

Skip the per-pair diff only when the change cannot possibly affect scoring: documentation, comment-only edits, test refactors that don't touch source, or scripts in the gitignored `scripts/` directory. Anything that touches `src/pd_matcher/` — even a constant — runs through the diff before merge.

## After scoring changes: rebuild the labeling queue

The review queue (`data/review.db`) is the SQLite database the labeling UI reads from. It is built by scoring a stratified MARC sample against the current matcher and writing the top candidates per record. **Once it's built, it doesn't update itself** — every pair in it carries the score and band assignment computed at build time.

After merging any branch that changes how the matcher scores (new pairings, new signals, new normalization rules, weight changes), the queue is stale relative to `main`. Pairs that *now* fall into the labeling bands aren't surfaced; pairs that *now* score below threshold are still in rotation. Labels captured against a stale queue are informative against the old matcher, not the current one.

```bash
pdm run pd-groundtruth build-queue --rebuild
```

`--rebuild` drops the existing `data/review.db` and re-runs scoring against the current pool and matcher. Vault verdicts are preserved automatically — every entry in `data/label_vault.jsonl` is pre-applied into the freshly-built queue, so pairs you've already labeled stay labeled.

**Batching.** The rebuild is the same cost regardless of how many scoring branches merged since the last one, so defer it until a cluster of related branches has all landed. The previous status snapshot in the resume-here memory tracks when a rebuild is pending; check it before labeling.

Skip the rebuild when the change cannot affect scoring (docs, tests, scripts, refactors).
