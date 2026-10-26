---
tags: [project, log]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: project-setup
---
# Vault log

Append a concise dated line after substantive vault work. Link the relevant session or knowledge note. For trivial exchanges, no entry is needed.

- 2026-10-05 — Vault initialised from the Obsidian Memory Starter; seeded decisions, open questions and sources; hooks configured. [[Projects/Sessions/2026-10-05 Vault setup]]
- 2026-10-05 — Concepts lecture (L1–L8) and learner profile added. [[Projects/Sessions/2026-10-05 Concepts lecture]]
- 2026-10-13 — Training pipeline built; first Ettin-400M run macro-F1 0.926 (raw data). [[Projects/Sessions/2026-10-13 Training pipeline]]
- 2026-10-13 — No-class-weight Ettin run: macro-F1 0.947 (beats weighted 0.926); deadline 13 Nov recorded. [[Projects/Sessions/2026-10-13 Training pipeline]]
- 2026-10-13 — Class-weight sweep done: no weights best (0.947); transformer default switched to none. [[Projects/Sessions/2026-10-13 Training pipeline]]
- 2026-10-23 — Split comparison (normalised data): 80/20 macro-F1 0.935 vs 90/10 0.936 — no meaningful difference; split decision pending. See PROGRESS.md.
- 2026-10-23 — Controlled check (normalised text, original split): macro-F1 0.943 vs raw 0.947, McNemar p 0.83 — preprocessing has no measurable effect. See PROGRESS.md.
- 2026-10-26 — Synthetic negatives (0.925) and label fixes (0.930) both hurt vs 0.947 baseline; Richie's preprocessing verified. [[Projects/Sessions/2026-10-26 Synthetic negatives and label fixes]]
- 2026-10-26 — Decision: keep baseline; synthetic + label-fix results written up as negative findings in report draft (c). [[Projects/Sessions/2026-10-26 Synthetic negatives and label fixes]]
- 2026-10-26 — Split fixed at 80/20 by owner; exp branches merged into main; LR sweep + 4-epoch run started.
- 2026-10-26 — Tuning: lr 1e-5 0.935, 3e-5 0.933, 4 ep 0.936 vs baseline 0.947 (none significant); final = lr 2e-5, 3 ep.
- 2026-10-26 — Final model trained: val macro-F1 0.940; submission.csv written (1,851 rows).
- 2026-10-26 — submission.csv (from ettin400m_final, val macro-F1 0.940) set as current submission at repo root.
- 2026-10-26 — Added src/evaluate.py (confusion/ROC/PR/calibration/per-class) after reviewing alumni repo; figures for ettin400m_final in docs/figures/.
- 2026-10-26 — Research: Ettin gaps + hybrid methods (docs/ettin_gaps_and_hybrids.md). Verified: averaging 9 Ettin runs' val probs = macro-F1 0.943 (< best single 0.947); 22 reviews wrong in all 9; 21 of 44 final-model errors at >0.99 confidence. Proposal (not decided): Ettin + DeBERTa-v3-large + fine-tuned SiEBERT average.
- 2026-10-26 — Verified: SiEBERT zero-shot vs Ettin final error correlation 0.59 (Ettin vs Ettin 0.69), 31 of 44 Ettin errors shared; Ettin+SiEBERT average 0.935 < Ettin 0.940. Ensemble case weakened; next: single DeBERTa-v3-large run first.
- 2026-10-26 — Verified: routing Ettin's uncertain reviews works (AUROC 0.86; top-100 holds 30/44 errors) but all models score 60–74% on them (chance ~50%); hard-only specialist judged weak (label noise). Proposal: DeBERTa as cascade specialist, routing size fixed on OOF.
- 2026-10-26 — Verified: 5 off-the-shelf review models (nlptown, Nav772 RoBERTa-5star, RashidNLP DeBERTa, LiYuan, SiEBERT) as cascade specialists on Ettin's top-30/60/100 uncertain val reviews: 53–73% acc vs Ettin 62–70%; best cascade 0.942 vs 0.940 (noise). No ambiguity-specialist model found in literature.
