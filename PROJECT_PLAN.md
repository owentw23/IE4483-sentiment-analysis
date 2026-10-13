# IE4483 Sentiment Analysis roadmap

Created: 2026-10-05. This file is the authoritative plan and sequence. Fixed decisions and the validation protocol are in [CLAUDE.md](CLAUDE.md).

## Goal and scope

- Goal: a learning-based binary sentiment classifier for product reviews, predictions for `test.json` in `submission.csv`, and a report answering questions (a)–(h) in `instruction2.png`. Stand out against fine-tuned RoBERTa/DeBERTa at student scale.
- In scope: the model arms and techniques listed under "Decisions made" in `CLAUDE.md`.
- Out of scope: anything listed there as dropped or rejected.
- Constraints: train only on `train.json` (pretrained weights are allowed: the brief names pretrained embeddings and BERT); hardware is an RTX 4060 Laptop (8 GB), Colab T4 as fallback. **Deadline: Friday 13 Nov 2026, 11:59 pm** (from the brief, pasted 2026-10-13).

## Stages and acceptance gates

| Stage | Work | Evidence needed to complete | State |
|---|---|---|---|
| 1 | Shared data/split module; TF-IDF and BiLSTM + GloVe baselines as scripts | Scripts reproduce the baseline table in `PROGRESS.md` on the fixed split | Planned |
| 2 | Ettin-400M (fallback 150M) fine-tune with class-weighted loss | Validation acc / macro-F1 / neg-F1 recorded; `submission.csv` written | Planned |
| 3 | DeBERTa-v3-large baseline, then optional arms (Ettin decoder, Mamba, Qwen3 embeddings, zero-shot) | Each arm's metrics on the same split | Planned |
| 4 | Synthetic-negatives ablation (with/without) | Instructor approval recorded; with/without metrics | Blocked on instructor |
| 5 | Report (a)–(h): SiEBERT error samples for (e); cleanlab / co-teaching / label smoothing for (h); zero-shot + domain-adaptive pretraining + SetFit for (g) | Complete report draft with all [TBD]s filled | Drafting (a–c) |

## Decision rules

- Every model uses the identical validation protocol in `CLAUDE.md` (single seed 42) and reports accuracy, macro-F1 and negative-class F1.
- Scope changes are decided by the project owner; record them in `CLAUDE.md` and [Scope decisions](<Knowledge/Projects/Scope decisions.md>) in the vault.

## Open planning questions

- `submission.csv` format and synthetic-data permission: both for the instructor (see `PROGRESS.md`).

For current progress and the immediate next action, read [PROGRESS.md](PROGRESS.md).
