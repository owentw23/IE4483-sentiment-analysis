---
tags: [project, decisions]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: project-conversation
---
# Scope decisions

The canonical list of **agreed** model and technique choices is the "Decisions made" section of the project-root `CLAUDE.md`. That section includes the dropped and rejected items, which should not be proposed again. Use this note for the *why* behind a decision, with a date and who decided. Keep proposals separate. See the [roadmap](../../PROJECT_PLAN.md) for sequencing.

## Agreed (recorded before 2026-10-05, decided by the project owner)

- Main model is Ettin-encoder-400M (fallback 150M) rather than ModernBERT-large. It has the same architecture and speed, plus open data, more sizes and a paired decoder twin. Detail: `docs/research_notes.md`.
- The baseline to beat is DeBERTa-v3-large.
- Techniques are limited to class-weighted loss and LLM-generated synthetic negatives (with/without ablation). R-Drop, model soups, multiple seeds and threshold tuning were dropped by the owner.

## Proposed

- None.
