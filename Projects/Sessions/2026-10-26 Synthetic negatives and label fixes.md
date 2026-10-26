---
tags: [project, session]
status: developing
created: 2026-10-26
modified: 2026-10-26
source: project-conversation
---
# 2026-10-26 — Synthetic negatives and label fixes

## Request

Owner reported the instructor approved synthetic data. Asked Claude to read the training reviews, write 800 negative reviews from personas, fix mislabelled training items, and retrain. Also asked to verify Richie's `PreProcessingPipeline/preprocess.py` was fully applied.

## Decisions

- **Agreed (implemented):** synthetic data and label fixes touch the training split only; validation labels never changed (owner's earlier "cheating" concern). Original 80/20 split for comparability with the 0.947 baseline.
- **Advised, owner accepted:** not 1:1 balance (class weights already showed balancing hurts); 800 targeted mostly at mixed and implicit negatives.

## Work and evidence

- Files: `src/cv_flag.py` (5-fold OOF predictions), `src/synthetic.py` (parse, format-match, dedup), `src/train.py` (`--extra-train`, `--label-fixes`), `data/synthetic/` (personas README, 9 batches, `synthetic_negatives.json`), `data/label_fixes.csv`.
- Measured (single seed each, McNemar vs baseline on the same 1,437 reviews): baseline 0.947; +800 synthetic 0.925 (p = 0.013); label fixes 0.930 (p = 0.036); both 0.931 (p = 0.036).
- Diagnosis (measured): TF-IDF detector separates synthetic from real negatives at 0.918 → style shortcut; missed negatives 15 → 30 across blunt and non-blunt reviews. Label-fix errors are mostly mild-complaint positives → labels follow a star-rating convention shared by validation.
- Richie's pipeline: `prepare_data()` output identical to `--normalize` (both splits, test); tokenizer identical to 400M; differences are max length 256 vs 512 and class weights he computes but does not apply.

## Next action

Owner to choose: same-voice synthetic data for both classes, detector-filtered rewrite, small dose, or report as a negative result. Then the split decision and `--epochs 4`.
