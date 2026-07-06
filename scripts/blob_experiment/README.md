# Blob matcher experiment (GH #128) — frozen snapshot, 2026-07-05

Measurement harnesses and reports for the field-free "blob" matcher prototype:
all descriptive text from each side concatenated, normalized with the
production pipeline, scored by token-overlap Jaccard (IDF-weighted and
unweighted variants). Full results and the decision trail live on
[#128](https://github.com/jpstroop/pd-matcher/issues/128).

Status: **parked, not implemented.** Verdict from the three measurement
rounds: worse than the weighted arm as a ranker (pool AUC 0.9982 vs 0.9999;
loses top-1 24-13), but a near-complete independent veto of the weighted
arm's false-positive tail (99% at blob < 0.5, 84% at < 0.3) plus ~8 genuine
field-bleed recoveries per ~1,100 matches. The title-cheat refinements were
measured and rejected (the same-author trap is whole/part confusion — #82's
signal, not a title-weighting problem). Kept on this branch in case a
refinement idea makes it worth wiring in as a third arm / audit-tier vote.

## Contents

| file | what |
|---|---|
| `blob_common.py` | field extraction, tokenization, blob-IDF, scoring primitives |
| `build_blob_idf.py` | builds the blob IDF table from the full CCE corpus (~2 min); output cache not committed |
| `blob_measure.py` | Phase 1: pair-level scoring of all registration-arm vault pairs |
| `report.txt` | Phase-1 report (AUC table, both tails, complementarity) |
| `cheat_measure.py` / `cheat_probe.py` | title-cheat variant sweep + same-author-trap mechanism probe |
| `cheat_report.txt` | title-cheat report (negative result + whole/part diagnosis) |
| `phase2.py` | Phase 2: ranking test over real retrieval pools (4.22M candidate-pairs) |
| `phase2_report.txt` / `timing.txt` | Phase-2 report and timing sample |
| `pairs.csv` | per-pair stock evidence + blob variant scores (Phase-1 snapshot) |

Run everything from the repo root with `pdm run python scripts/blob_experiment/<script>`.
`build_blob_idf.py` must run first to regenerate the IDF cache; the scripts
expect the caches (`caches/cce.lmdb`, calibrator) and `data/training/label_vault.jsonl`
to exist. Scripts are frozen audit snapshots of the measurement, not maintained code.
