# IE4483 Sentiment Analysis progress and handoff

Updated: 2026-10-26. This file is the authoritative current implementation status. The sequence and gates are in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Current state

- **As of 2026-10-26: report drafting starts.** Stage 2 (Ettin) done; Stage 3 partly done (Ettin decoder twin, cascade experiments); Stage 4 done (synthetic negatives measured, rejected).
- Submitted: repo-root `submission.csv` = Ettin-400M encoder `ettin400m_final` (val acc 0.969, macro-F1 0.940). Owner decision: keep it; the Ettin + Qwen3-8B cascade (0.946, not significant) is for report (e) only.
- Report material for the drafter:
  - (a)–(c): `docs/report_draft_abc.md`; full specs, settings and reasons for (b)/(c): `docs/model_architecture_bc.md`.
  - (e): `docs/report_draft_e.md` (strength case test id 276, weakness case id 82, limit case id 1479).
  - Ettin's gaps, hybrid/cascade experiments, benchmark context (useful for (g)/(h)): `docs/ettin_gaps_and_hybrids.md`.
  - Evaluation figures: `docs/figures/ettin400m_final/`. Team summary page: https://claude.ai/artifact/GPY3XM9ySGpu6TvySQYPgz (owner shares it).
- Not yet done: DeBERTa-v3-large baseline; TF-IDF and BiLSTM + GloVe as committed scripts (TF-IDF numbers below are from exploratory runs); report (d), (f), (g), (h).
- Open with instructor: `submission.csv` column format; whether an LLM step counts under "train only on the training set".
- Failed or blocked: none.

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
| **Cascade** Ettin final + Qwen3-8B (4-bit, 8 nearest training reviews in the prompt) when Ettin confidence < 0.99 (threshold chosen on 5-fold OOF) — `cascade_qwen3-8b` (2026-10-26; 59 val reviews routed; fixed 13 / broke 8 vs Ettin final, McNemar p = 0.38; test: 69 routed, 26 changed, `outputs/runs/cascade_qwen3-8b/submission.csv`). Similarity vote 0.937 and embedding LogReg 0.930 as specialists did not help (`cascade_embed`). Details: `docs/ettin_gaps_and_hybrids.md` | 0.973 | 0.946 | 0.907 |
| Ettin **decoder**-400M fine-tuned, no weights (lr 2e-5, 3 ep, last-token pooling, best ep 2, 13.2 min) — `ettin_dec400m_lr2e-5` (2026-10-26; vs encoder `ettin400m_final` 0.940, McNemar p = 0.061) | 0.962 | 0.924 | 0.870 |

### Split comparison with groupmate preprocessing (`--normalize`, no weights, lr 2e-5, 3 ep) — 2026-10-23

| Run | Split (train / val) | Acc | Macro-F1 | Neg-F1 | Neg P/R | Best ep | Per-epoch macro-F1 | Min |
|---|---|---|---|---|---|---|---|---|
| `ettin400m_8020_norm` | 80/20 (5,745 / 1,437) | 0.968 | 0.935 | 0.889 | 0.902 / 0.876 | 2 | 0.932 / 0.935 / 0.932 | 12.7 |
| `ettin400m_9010_norm` | 90/10 (6,463 / 719) | 0.969 | 0.936 | 0.889 | 0.946 / 0.838 | 1 | 0.936 / 0.928 / 0.931 | 13.5 |

Gap 0.0005 macro-F1 — well inside noise (±1 / ±1.3 pt), so neither split is better on score. Both are ~1.2 pt below the raw-data reference (0.947), but the normalised 80/20 val set shares only 1,143 of 1,437 reviews with the old one, so that drop is not attributable to preprocessing alone. **Owner chose 80/20 (2026-10-26)** — recorded in `CLAUDE.md`.

Controlled check (`ettin400m_8020_normtext_origsplit`, `--normalize-text-only`: original 5,748 / 1,437 split, whitespace normalised afterwards, so preprocessing is the only change): acc 0.972, macro-F1 0.943, neg-F1 0.903, neg P/R 0.901 / 0.905, best ep 3 (0.919 / 0.935 / 0.943), 12.6 min. vs raw 0.947: 41 vs 39 errors on the same 1,437 reviews (29 shared, 10 only raw, 12 only normalised; McNemar exact p = 0.83). **Whitespace normalisation neither helps nor hurts**; the earlier 1.2-pt drop was the changed validation set.

### Synthetic negatives and label cleaning (original 80/20 split, no weights, lr 2e-5, 3 ep) — 2026-10-26

Instructor approved synthetic data. 800 negatives written by Claude from 16 personas (`data/synthetic/`, 330 mixed / 107 implicit / 363 plain, formatting matched to real negatives, max TF-IDF cosine to any real review 0.34). Label fixes from 5-fold out-of-fold Ettin predictions on the training split (`src/cv_flag.py`; OOF acc 0.975): 146 flagged, each read by hand → 9 relabelled to positive, 23 to negative, 64 mixed/off-topic dropped, 50 kept (`data/label_fixes.csv`). Validation labels never touched. McNemar = paired test vs the baseline on the same 1,437 reviews.

| Run | Train rows | Acc | Macro-F1 | Neg-F1 | Neg P/R | Missed neg / missed pos | vs baseline (only base wrong / only run wrong, p) |
|---|---|---|---|---|---|---|---|
| baseline `ettin400m_lr2e-5_noweights` | 5,748 | 0.973 | **0.947** | 0.909 | 0.890 / 0.929 | 15 / 24 | — |
| `ettin400m_syn800` (+800 synthetic) | 6,548 | 0.963 | 0.925 | 0.872 | 0.887 / 0.857 | 30 / 23 | 7 / 21, p = 0.013 |
| `ettin400m_labelfix` | 5,684 | 0.965 | 0.930 | 0.882 | 0.860 / 0.905 | 20 / 31 | 8 / 20, p = 0.036 |
| `ettin400m_labelfix_syn800` | 6,484 | 0.965 | 0.931 | 0.882 | 0.857 / 0.909 | 19 / 32 | 8 / 20, p = 0.036 |

**Both techniques hurt; the plain baseline stays best.** Single training seed each.
- Synthetic: a TF-IDF + LogReg detector tells the synthetic negatives from real ones at 0.918 (5-fold), on style cues (contractions, present-tense stories, "two weeks", wedding/dress personas vs real "was", "item", "product", "Amazon", "you"). Half the negative class then carried a non-real style, so real-customer style became a cue for positive: missed negatives doubled (15 → 30), both with blunt words (2 → 7 of 98) and without (13 → 23 of 112). The synthetic set has fewer blunt words than real negatives (19% vs 38%), so "too blunt" is ruled out.
- Label fixes: the "mislabels" are mostly the dataset's star-rating convention (mild complaints labelled positive, praise-then-flaw labelled negative), which validation (and presumably test) labels share. Relabelling to the text's sentiment and dropping mixed cases moved the model off that convention: new errors are mostly mild-complaint positives (missed positives 24 → 31).

Each Ettin run: ~11.9 min for 3 epochs, peak GPU 5.6 GB, 395.8M params. Weighted: neg P/R 0.858/0.891. Unweighted: neg P/R 0.890/0.929, so it is better on **both** precision and recall (not just a shifted boundary). Single seed each, so part of the gap may be seed variance. Unweighted was still improving at epoch 3 (0.926 → 0.942 → 0.947).

## Next action

- **2026-10-26 owner decision:** submission stays plain Ettin (`ettin400m_final`); the Ettin + Qwen3-8B cascade (`cascade_qwen3-8b`, val 0.946) is used for report (e) and as the base for further work, not submitted. Draft of (e): `docs/report_draft_e.md`.
- **2026-10-26 evaluation figures:** `src/evaluate.py --run <run>` turns a run's `val_predictions.csv` into confusion matrix, ROC, negative-class PR curve, reliability diagram, `per_class.csv` and `summary.json` under `docs/figures/<run>/` (CPU only). For `ettin400m_final`: confusion [[192, 18], [26, 1201]], ROC-AUC 0.985, negative-class PR-AUC 0.926, Brier 0.029, ECE 0.028; 97.6% of reviews get P > 0.95 or < 0.05, and the mid-range bins hold only 1–5 reviews each, so calibration there is not measurable. Idea taken from the alumni repo (Tonymnugget/IE4483-Sentiment-Analysis-Project); their reported scores use a non-deduplicated 10% split where 35 of 741 validation reviews also appear in train.
- **2026-10-26 final model:** `ettin400m_final` (lr 2e-5, 3 ep, original protocol, `--save-model --predict-test`): val acc 0.969, macro-F1 **0.940**, neg-F1 0.897, neg P/R 0.881 / 0.914, best epoch 2 (0.921 / 0.940 / 0.937), 13.0 min. Same recipe as the 0.947 run; the gap is GPU non-determinism (98.3% of validation predictions agree). `outputs/runs/ettin400m_final/submission.csv`: 1,851 rows, columns `id,sentiments` (format still unconfirmed with instructor), 13.9% predicted negative (train 14.6%). Model weights in `outputs/runs/ettin400m_final/model` (1.6 GB, not in git). **Current submission (owner, 2026-10-26):** copied to repo-root `submission.csv` from `ettin400m_final`. Can be replaced later (e.g. a `--normalize` retrain to match report (a), or a better method); rerun `src/train.py --run-name <name> --save-model --predict-test` and copy that run's `submission.csv` over the root file.

- **2026-10-26:** split fixed at 80/20; both experiment branches merged into `main`. Tuning done (original protocol, McNemar vs baseline): lr 1e-5 0.935 (best ep 2, p = 0.13), lr 3e-5 0.933 (ep 1 only 0.901, p = 0.11), lr 2e-5 × 4 epochs 0.936 (peaked ep 2, p = 0.29). **Final settings: lr 2e-5, 3 epochs (baseline, 0.947).** Eleven configurations on this split span 0.925–0.947, so 0.947 is likely the lucky end of single-seed variation; expect ~0.935–0.945. Report (c) tables filled. Next: final model with `--save-model --predict-test` → `submission.csv`; TF-IDF/BiLSTM scripts for (f); report (d)–(h).

0. **2026-10-26:** synthetic negatives and label cleaning measured — both hurt (table above). **Owner decision: keep the baseline (original training data) and report both as negative findings** — written up in `docs/report_draft_abc.md` (c), "Training-data changes we tried and rejected"; the label-cleaning result also feeds (h). Richie's `preprocess.py` verified step by step: identical splits to `--normalize`; only deliberate differences are max length 256 vs our 512 and his unused class weights.
0. **2026-10-23:** run the 80/20 vs 90/10 split comparison with the groupmate's preprocessing on branch `exp/split-comparison` — steps in `docs/split_comparison.md`. Groupmate's `PreProcessingPipeline/submission.csv` is a placeholder (all 0) — do not submit it.

1. Class-weight question settled (owner chose the fixed sweep): **no weights** is best (0.947 vs 0.936 / 0.938 / 0.926 for neg weight 1.5 / 2.0 / balanced). `src/train.py` now defaults to `--class-weights none`.
2. Next: `--epochs 4` (best epoch was the last one), then the LR sweep `src/sweep.py --tag lr --lr 1e-5 3e-5` (`docs/training_guide.md` §3). Rerun everything on the groupmate's cleaned file once it arrives (`--train-file`).

## Open questions (owner: instructor)

- `submission.csv` column format is unspecified.
- ~~Whether LLM-generated synthetic negatives are allowed~~ — instructor said yes (reported by owner 2026-10-26).

## Recent evidence

| Date | Claim | Evidence / check | Limits |
|---|---|---|---|
| 2026-10-13 | Training pipeline works on 8 GB GPU | `src/sanity_check.py` PASS (init loss 0.555, overfit 16 reviews); fp32 AdamW OOM-spills (7.9 GB), 8-bit AdamW peak 5.6 GB | Single seed; raw (uncleaned) data |
| 2026-10-05 | Obsidian vault memory and hooks set up | `Knowledge/`, `scripts/vault-hook.cjs`, `.claude/settings.local.json` | Hooks pipe-tested only; a live event is confirmed in the next session |

Historical rationale and session records live under [Knowledge/](Knowledge/START-HERE.md). Update this handoff after every meaningful milestone; keep it concise.
