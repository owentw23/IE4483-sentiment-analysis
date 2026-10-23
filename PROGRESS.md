# IE4483 Sentiment Analysis progress and handoff

Updated: 2026-10-13. This file is the authoritative current implementation status. The sequence and gates are in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Current state

- Stage: 2 (Ettin fine-tune) pipeline built and first run done; Stage 1 TF-IDF/BiLSTM scripts not yet written. Groupmate is cleaning the data (2026-10-13).
- Completed and verified: exploratory baselines on the fixed validation split (table below); data facts and token-length statistics recorded in `CLAUDE.md`; report draft for (a)–(c) in `docs/report_draft_abc.md`.
- In progress: hyperparameter tuning plan in `docs/training_guide.md` §3 (next step: learning-rate sweep).
- Failed or blocked: none.
- Not yet done: no saved final model, no `submission.csv`, no TF-IDF/BiLSTM scripts; uncommitted code in `src/`. Baseline numbers below came from exploratory runs, not committed scripts, so reproduce them when Stage 1 lands.

## Validation results (dedup + stratified 80/20, seed 42, 1,437 reviews)

| Model | Acc | Macro-F1 | Neg-F1 |
|---|---|---|---|
| Majority class | 0.854 | 0.46 | 0.00 |
| TF-IDF (1–2 gram) + LogReg | 0.911 | 0.766 | 0.582 |
| TF-IDF + LogReg, class_weight=balanced | 0.920 | 0.841 | 0.728 |
| SiEBERT zero-shot | 0.958 | 0.921 | 0.866 |
| Ettin-400M fine-tuned, class-weighted (raw data, lr 2e-5, 3 ep, best ep 2) — `ettin400m_lr2e-5_default` | 0.962 | 0.926 | 0.874 |
| **Ettin-400M fine-tuned, no class weights** (same otherwise, best ep 3) — `ettin400m_lr2e-5_noweights` | **0.973** | **0.947** | **0.909** |
| Ettin-400M, neg weight 1.5 / pos 1.0 (best ep 3) — `cwsweep_negw1.5` | 0.968 | 0.936 | 0.891 |
| Ettin-400M, neg weight 2.0 / pos 1.0 (best ep 3) — `cwsweep_negw2.0` | 0.969 | 0.938 | 0.895 |

Each Ettin run: ~11.9 min for 3 epochs, peak GPU 5.6 GB, 395.8M params. Weighted: neg P/R 0.858/0.891. Unweighted: neg P/R 0.890/0.929, so it is better on **both** precision and recall (not just a shifted boundary). Single seed each, so part of the gap may be seed variance. Unweighted was still improving at epoch 3 (0.926 → 0.942 → 0.947).

## Next action

0. **Now (2026-10-23):** run the 80/20 vs 90/10 split comparison with the groupmate's preprocessing on branch `exp/split-comparison` — steps in `docs/split_comparison.md`. Groupmate's `PreProcessingPipeline/submission.csv` is a placeholder (all 0) — do not submit it.

1. Class-weight question settled (owner chose the fixed sweep): **no weights** is best (0.947 vs 0.936 / 0.938 / 0.926 for neg weight 1.5 / 2.0 / balanced). `src/train.py` now defaults to `--class-weights none`.
2. Next: `--epochs 4` (best epoch was the last one), then the LR sweep `src/sweep.py --tag lr --lr 1e-5 3e-5` (`docs/training_guide.md` §3). Rerun everything on the groupmate's cleaned file once it arrives (`--train-file`).

## Open questions (owner: instructor)

- `submission.csv` column format is unspecified.
- Whether LLM-generated synthetic negatives are allowed (the brief says to train only on the training set).

## Recent evidence

| Date | Claim | Evidence / check | Limits |
|---|---|---|---|
| 2026-10-13 | Training pipeline works on 8 GB GPU | `src/sanity_check.py` PASS (init loss 0.555, overfit 16 reviews); fp32 AdamW OOM-spills (7.9 GB), 8-bit AdamW peak 5.6 GB | Single seed; raw (uncleaned) data |
| 2026-10-05 | Obsidian vault memory and hooks set up | `Knowledge/`, `scripts/vault-hook.cjs`, `.claude/settings.local.json` | Hooks pipe-tested only; a live event is confirmed in the next session |

Historical rationale and session records live under [Knowledge/](Knowledge/START-HERE.md). Update this handoff after every meaningful milestone; keep it concise.
