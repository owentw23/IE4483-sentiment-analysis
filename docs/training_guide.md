# Training guide: questions (b), (c), (d) and how to tune

Code lives in `src/`. Every number below marked *measured* comes from `outputs/results.csv` on the RTX 4060 Laptop (8 GB).

## 0. Quick start

```bash
.venv/Scripts/python src/data.py                     # split + class weights (5,748 train / 1,437 val)
.venv/Scripts/python src/sanity_check.py             # initial loss ~0.69, overfit one batch -> PASS
.venv/Scripts/python src/train.py --run-name NAME    # one run with the planned defaults
.venv/Scripts/python src/sweep.py --show             # all runs, best macro-F1 first
```

| File | Role | Report question |
|---|---|---|
| `src/data.py` | Load, deduplicate, stratified 80/20 split (seed 42), class weights | (a), (b) |
| `src/sanity_check.py` | Initial-loss and overfit-one-batch checks (lecture L5) | (b) evidence |
| `src/train.py` | One fine-tuning run; writes metrics, history, validation predictions, ledger row | (b), (c) |
| `src/sweep.py` | Grid search; each run in a fresh process | (c) |
| `src/predict.py` | Saved model → `submission.csv` (+ probabilities for error analysis) | (d), (e) |

## 1. Question (b): what the pipeline does, mapped to the brief

| The brief asks for | What we have | Where |
|---|---|---|
| Model architecture figure | Ettin-encoder-400M (395.8M params, *measured*) → masked mean pooling → dense+GELU+LayerNorm head → Linear 1024→2. The figure with shapes is in `docs/report_draft_abc.md` (b) | `AutoModelForSequenceClassification` |
| Input and output dimensions | Input: token ids + attention mask `[B, L]`, L ≤ 512, dynamic padding. Output: logits `[B, 2]` → softmax → argmax | `TextDataset`, `DataCollatorWithPadding` |
| Loss function | Class-weighted cross-entropy, weights neg **3.426** / pos **0.585** from the training split | `WeightedTrainer.compute_loss` |
| Training strategy | AdamW (8-bit) · lr 2e-5 · linear warm-up 10% then linear decay · batch 8 × 2 accumulation = 16 · 3 epochs · bf16 · gradient checkpointing · evaluate each epoch, keep the best validation macro-F1 checkpoint | `TrainingArguments` in `train.py` |
| Code and how to run it | §0 above plus `README.md` | — |
| Reproducibility | Fixed seed 42 (`set_seed` + `TrainingArguments.seed`), deterministic split (`random_state=42`), pinned `requirements.txt`, single run per config (the brief accepts "fixing random seed") | `train.py`, `data.py` |

Evidence to include in (b):
- Sanity checks (*measured*): initial loss 0.555 (vs ln 2 = 0.693) and a 16-review batch overfitted to ~0 within 10 steps. The pipeline learns.
- Memory (*measured*): plain fp32 AdamW filled the 8 GB card (7.9 GB) and stalled; 8-bit AdamW + checkpointing peaked around 6 GB.
- Training curves: `outputs/runs/<run>/history.json` holds loss per 25 steps and validation metrics per epoch. Plot them for the report.

## 2. Question (c): every setting, how it was chosen, and why

| Setting | Flag | Default | How chosen | Why |
|---|---|---|---|---|
| Learning rate | `--lr` | 2e-5 | **Swept** {1e-5, 2e-5, 3e-5} on validation macro-F1 | Most sensitive hyperparameter. Too high wrecks pretrained weights; too low underfits in 3 epochs. 1e-5–5e-5 is the fine-tuning range (Devlin 2019, Mosbach 2021) |
| Epochs | `--epochs` | 3 | Literature (2–4) + **checkpoint selection** picks the best epoch | Small dataset → overfits quickly; validation decides |
| Warm-up | `--warmup-ratio` | 0.1 | Literature | Random head + unsettled Adam moments at the start; protects pretrained weights |
| Schedule | `--scheduler` | linear | Literature (BERT recipe) | Large steps early, small steps late |
| Weight decay | `--weight-decay` | 0.01 | Literature (AdamW default for BERT-style) | Mild regularisation |
| Effective batch | `--batch-size` × `--grad-accum` | 8 × 2 = 16 | **Hardware** + literature | Batch 8 fits 8 GB; accumulation gives a standard batch of 16 (~360 updates/epoch) |
| Max length | `--max-length` | 512 | **Data**: token stats (p99 = 346, max 1,293) | Covers 99.67% of reviews whole; dynamic padding keeps it cheap |
| Class weights | `--class-weights` / `--neg-weight` | **none** | **Swept** (*measured*): none 0.947, neg 2.0 0.938, neg 1.5 0.936, balanced (3.43/0.585) 0.926 | Helped TF-IDF (+0.075) but hurt the fine-tuned transformer. Without weights, negative precision **and** recall were both higher, so the weights degraded learning rather than just shifting the boundary |
| Optimizer precision | `--optim` | adamw_bnb_8bit | **Hardware** (*measured*: fp32 AdamW doesn't fit) | 8-bit states cut optimizer memory 4× with negligible loss |
| Mixed precision | `--no-bf16` to disable | bf16 on | **Hardware** | Halves activation memory, faster on tensor cores |
| Gradient checkpointing | `--no-grad-ckpt` to disable | on | **Hardware** | Trades ~25% speed for activation memory |
| Selection metric | (fixed in code) | macro-F1 | Data: 85% majority class | Accuracy hides failure on negatives |
| Seed | `--seed` | 42 | Fixed | Reproducibility |

## 3. Finding the optimal settings: what to change, in order

Run these steps in order. Each step fixes one setting before the next. *Measured*: one Ettin-400M run (3 epochs) = **11.8 min**, peak 5.6 GB. Steps 1–6 ≈ 10–12 runs ≈ **2–2.5 hours** of GPU time (DeBERTa-v3-large probably ~1.5× slower; Ettin-150M ~2× faster; both are estimates).

First full run (*measured*, raw data, defaults, lr 2e-5): validation macro-F1 per epoch 0.924 → **0.926** → 0.924; validation loss 0.387 → 0.313 → 0.341; training loss ended at ~0.01. Epoch 2 was selected. The rise in validation loss at epoch 3, while training loss approaches 0, is mild overfitting, and checkpoint selection handled it.

| Step | Command | Runs | What you learn | Decision rule |
|---|---|---|---|---|
| 0 | `src/sanity_check.py` | 1 | Pipeline learns | Must PASS |
| 1. **Learning rate** | `src/sweep.py --tag lr --lr 1e-5 3e-5` (2e-5 is already done: `ettin400m_lr2e-5_default`) | 2 (~24 min) | Best η | Highest val macro-F1. If 3e-5 wins, also try 5e-5 |
| 2. Epochs | Look at `best_epoch` in `--show` | 0 | Under- or overtraining | If best = 3 (last), rerun the winner with `--epochs 4`. If best = 1, epochs are fine (selection handles it) |
| 3. Batch (optional) | `src/sweep.py --tag bs --lr <best> --grad-accum 2 4` | 2 | Effective batch 16 vs 32 | Keep 16 unless 32 is > 1 point better |
| 4. Ablation for the report | `src/sweep.py --tag cw --lr <best> --class-weights balanced none` | 1 new | Effect of class weights | Report both; keep balanced unless worse |
| 5. Baseline to beat | `src/sweep.py --tag deberta --model microsoft/deberta-v3-large --lr 1e-5 2e-5` | 2 | Industry-standard comparison | Same protocol; higher val macro-F1 is the submission model |
| 6. Cost arm for (f) | `src/sweep.py --tag e150 --model jhu-clsp/ettin-encoder-150m --lr <best> 5e-5` | 2 | Accuracy vs size/time | Report alongside time and memory |
| Final | `src/train.py --run-name FINAL --lr <best> ... --save-model --predict-test` | 1 | The submission model | — |

Rules while tuning:
- **Differences < ~1 point (0.01) macro-F1 are ties** at n = 1,437 with ~210 negatives. Prefer the default or the cheaper setting on ties.
- Change one thing at a time. Never look at test predictions to choose settings.
- Already decided **not** to tune (owner's call): decision threshold, multiple seeds, model soups, R-Drop.
- Smoke-test any new command first with `--limit 64 --epochs 1` (about 30 s).

## 4. Question (d): producing `submission.csv`

1. Final run: `src/train.py --run-name FINAL --lr <best> --save-model --predict-test` writes `outputs/runs/FINAL/submission.csv`.
2. Or later from the saved model: `src/predict.py --model-dir outputs/runs/FINAL/model`. This writes `submission.csv` plus `submission_with_probs.csv` (text + P(positive), for question e).
3. Format: one row per test review in **original order**, columns `id, sentiments` (0/1). **Unconfirmed with the instructor.** If they want another layout, change `COLUMNS` in `src/predict.py`.
4. Check: 1,851 rows, values only 0/1, and the predicted negative share should be near the validation rate (~15%).

## 5. Handing over cleaned data (for the groupmate)

The scripts take any cleaned file: `--train-file data/train_clean.json --test-file data/test_clean.json` (JSON or CSV).

Requirements so results stay valid:
1. **Same columns**: `reviews` (text) and `sentiments` (0/1, training only).
2. **Test file keeps all 1,851 rows in the original order**. No dropping or deduplicating test rows; the submission must line up.
3. **Light cleaning only for transformers.** Fixing encoding junk, HTML tags and whitespace is fine. Do **not** lowercase, remove stop-words or punctuation, or stem. Ettin's tokenizer uses case, "not"/"but" and "!!" (lecture L3).
4. **Apply identical cleaning to train and test.**
5. If cleaning changes which training rows exist, the validation split changes. Then **every** model (TF-IDF, Ettin, DeBERTa…) must be rerun on the same cleaned file, or the comparison breaks.
