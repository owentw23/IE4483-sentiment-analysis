---
tags: [project, session]
status: developing
created: 2026-10-13
modified: 2026-10-13
source: project-conversation
---
# 2026-10-13 — Training pipeline for (b), (c), (d)

## Request

Prepare (b), (c) and (d): explain them in detail and show what to change to find the best settings. A groupmate is cleaning the data in parallel.

## Decisions

- **Agreed (implemented):** Hugging Face `Trainer` with a class-weighted CE subclass; 8-bit AdamW + bf16 + gradient checkpointing; each sweep run in its own process; results logged to `outputs/results.csv`.
- **Proposed:** the tuning order in `docs/training_guide.md` §3. Learning rate first, then epochs, optional batch size, the class-weight ablation, DeBERTa, then Ettin-150M.
- **Fact corrected:** the GPU is an RTX 4060 Laptop (8 GB), not a 5060. CLAUDE.md, README, L6 and the report draft are updated.

## Work and evidence

- Files: `src/data.py`, `src/train.py`, `src/sweep.py`, `src/predict.py`, `src/sanity_check.py`, `requirements.txt`, `docs/training_guide.md`, report draft (b) code section.
- Checks (measured): sanity PASS (initial loss 0.555; 16 reviews overfitted within 10 steps). fp32 AdamW filled VRAM at 7.9 GB and stalled. Full default run: macro-F1 0.924 / **0.926** / 0.924 over 3 epochs, acc 0.962, neg-F1 0.874, 11.8 min, peak 5.6 GB.
- Limits: raw, uncleaned data; single seed; `predict.py` not yet run (no saved model).

## Next action

LR sweep `src/sweep.py --tag lr --lr 1e-5 3e-5`. Rerun on the cleaned data when it arrives.
