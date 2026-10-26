---
tags: [project, session]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: project-conversation
---
# 2026-10-05 — Concepts lecture before implementation

## Request

The owner asked for a full lecture on the concepts needed before any implementation: how to answer (a)–(h) and how to choose the model and method. It builds on the unread EE4483 backprop lecture, with resources found via Perplexity.

## Decisions

- **Agreed:** implementation (Stage 1) waits until the owner has studied the lecture.
- **Proposed (in L7/L8, not yet agreed):**
  - Treat DeBERTa vs Ettin as decided purely by validation macro-F1, because their GLUE scores are within noise (91.1 vs 90.8).
  - For (e), use the 87 test reviews that also appear in training (labels known), or unambiguous test reviews, plus validation error statistics.

## Work and evidence

- Files: [[MOC - ML Training Lecture]] plus modules L1–L8 under `Resources/Lecture/`; [[Areas/Learner profile]].
- Sources: the EE4483 PDF (read in full via `pdftotext`); external URLs gathered with Perplexity and listed per module.
- Limits: the external pages were not all opened in full. Training times and memory figures in L3/L6 are estimates until measured.

## Next action

The owner studies L1–L8. Then start Stage 1 (data module plus TF-IDF and BiLSTM scripts), see the [roadmap](../../../PROJECT_PLAN.md).
