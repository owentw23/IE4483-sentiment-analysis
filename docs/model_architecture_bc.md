# Model architecture, specs and design choices — reference for report questions (b) and (c)

Updated 2026-10-26. This is the single reference for whoever drafts (b) and (c): what the model is, every setting,
and why each choice was made, with the measured evidence. All numbers are on the same validation split unless stated:
`train.json` deduplicated (7,401 → 7,185 reviews), stratified 80/20 split with `random_state=42` → 5,748 train /
1,437 validation (210 negative). Main metric: macro-F1 (accuracy and negative-class F1 also reported).

**Submitted model:** Ettin-400M encoder, fully fine-tuned (`ettin400m_final`): validation accuracy 0.969,
macro-F1 0.940, negative-class F1 0.897. Its predictions are the repo-root `submission.csv`.

Contents: 1. Why this model · 2. Architecture specs (b) · 3. Loss and training (b) · 4. Reproducibility (b) ·
5. Code and how to run (b) · 6. How every parameter was chosen (c) · 7. Variants we built and tested ·
8. Results summary · 9. References

---

## 1. Why this model

**Choice: `jhu-clsp/ettin-encoder-400m`, a bidirectional transformer encoder with a classification head, fine-tuned
end to end.**

| Reason | Detail |
|---|---|
| Fine-tuned encoders are the standard for this task | On labelled sentiment data with thousands of examples, fine-tuned encoders (BERT, RoBERTa, DeBERTa) beat prompted LLMs (Zhang et al. 2023; Bucher & Martini 2024) |
| Modern architecture | Ettin uses the ModernBERT design (2024): rotary positions, alternating local/global attention, GeGLU, pre-norm. GLUE 90.8 for the 400M model vs 90.4 for ModernBERT-large and 88.9 for RoBERTa-large |
| Open, documented training data | Pretrained on ~2T tokens of fully open data with a published recipe, so its training data can be described in the report |
| Paired decoder twin | Ettin ships an encoder and a decoder trained on identical data with an identical recipe; only the objective differs. This allowed a controlled encoder-vs-decoder comparison (section 7) |
| Fits our hardware | 400M is the largest Ettin encoder that trains on an 8 GB laptop GPU (with 8-bit AdamW, bf16, gradient checkpointing; peak 5.6 GB). The 1B model does not fit |

**Alternatives considered**

| Model | Role in the project | Why not the main model |
|---|---|---|
| TF-IDF + logistic regression | Classical baseline | Loses word order and context; macro-F1 0.841 at best |
| BiLSTM + GloVe | Classic RNN baseline | Static word vectors; expected well below transformers (script still to be written) |
| SiEBERT (RoBERTa-large tuned on 15 sentiment datasets) | Zero-shot reference | Not trained on our labels; 0.921 zero-shot. Its prior training uses outside labelled data |
| DeBERTa-v3-large | Industry-standard baseline to beat | GLUE 91.4 (slightly above Ettin) but no open data, no decoder twin, slower; not yet run |
| Ettin decoder-400M | Controlled comparison | Tested: 0.924 vs encoder 0.940 (section 7) |
| Decoder LLMs (7–8B) | Tested only as a helper for uncertain reviews | Too large to fine-tune on 8 GB; LoRA on 7B models was ruled out |

---

## 2. Architecture specs (question b)

Verified from the saved model's `config.json` and module structure (`outputs/runs/ettin400m_final/model`).

| Component | Value |
|---|---|
| Model type | `modernbert` (Ettin encoder, 400M) |
| Total parameters | 395,833,346 (all trained) |
| — token embeddings | 51,577,856 (50,368 × 1,024) |
| — 28 transformer layers | 343,202,816 |
| — classification head | 1,051,650 |
| Tokenizer | BPE, vocabulary 50,368, case-sensitive, special tokens `[CLS]`, `[SEP]`, `[PAD]` |
| Hidden size | 1,024 |
| Layers | 28 |
| Attention heads | 16 per layer (64 dimensions each) |
| Attention pattern | Local sliding window of 128 tokens; every 3rd layer attends globally |
| Position encoding | Rotary position embeddings (RoPE), θ = 160,000 |
| Feed-forward | GeGLU: 1,024 → 2 × 2,624 (gate and value) → 2,624 → 1,024 |
| Normalisation | Pre-LayerNorm (ε = 1e-5); no bias terms in linear layers or norms |
| Max context | 7,999 positions supported; we use 512 |
| Dropout | 0 everywhere (Ettin default) |
| Pooling | Masked mean over all real tokens (`classifier_pooling = "mean"`) |
| Head | Dense 1,024 → 1,024 (no bias) → GELU → LayerNorm → Linear 1,024 → 2 (with bias) |
| Output | 2 logits → softmax → P(negative), P(positive); label = argmax |
| Labels | 0 = negative, 1 = positive |

**Architecture figure (to draw for the report)**

```
Review text (string)
  │  Tokenizer: BPE, adds [CLS] … [SEP], truncates to 512
  ▼
input_ids, attention_mask                        [B, L]          L ≤ 512 (median review 54 tokens)
  │  Token embedding + LayerNorm
  ▼
token vectors                                    [B, L, 1024]
  │  28 × encoder layer:
  │     x = x + Attention(LayerNorm(x))          16 heads, RoPE; 128-token local window,
  │                                              global attention in every 3rd layer
  │     x = x + GeGLU-MLP(LayerNorm(x))          1024 → 2×2624 → 2624 → 1024
  ▼
contextual token vectors                         [B, L, 1024]
  │  Final LayerNorm, masked mean pooling over real tokens
  ▼
review vector                                    [B, 1024]
  │  Head: Dense 1024→1024 → GELU → LayerNorm
  │  Classifier: Linear 1024→2
  ▼
logits                                           [B, 2]
  │  Softmax
  ▼
P(negative), P(positive)                         [B, 2]   → predicted label = argmax
```

**What each part does (plain language)**
1. **Tokenizer** splits text into subword pieces ("disappointing" → "dis", "appoint", "ing"), so any word, typo or
   emoji gets a representation. No lowercasing, stop-word removal or stemming: capitals ("TERRIBLE"), "not" and "but"
   carry sentiment.
2. **Embedding** maps each token ID to a 1,024-number vector learned in pretraining.
3. **Self-attention** lets every token look at the others and mix in their information, so "good" in "not good" gets
   a different vector than in "very good". Most layers look 64 tokens either side (cheap); every 3rd layer looks at the
   whole review. Our reviews (median 54 tokens) mostly fit inside one local window anyway.
4. **GeGLU feed-forward** transforms each token vector; the gate decides which features pass.
5. **Pre-LayerNorm and residual connections** keep training stable across 28 layers.
6. **Mean pooling** averages all token vectors into one review vector, so every word contributes rather than only the
   first token.
7. **Head and softmax** turn the review vector into two class scores and then probabilities.

---

## 3. Loss and training strategy (question b)

**Loss: standard cross-entropy, L = −log p(true class).** No class weights in the final model. Balanced class weights
(negative 3.43, positive 0.585) were tested and lowered macro-F1 (section 6), so they are only used for the TF-IDF
baseline and reported as an ablation.

| Setting | Value (from `outputs/runs/ettin400m_final/config.json`) |
|---|---|
| Fine-tuning | Full: all 395.8M parameters plus the new head |
| Optimizer | AdamW, 8-bit states (`bitsandbytes`, `adamw_bnb_8bit`) |
| Learning rate | 2e-5 peak |
| Schedule | Linear warm-up over the first 10% of steps, then linear decay to 0 |
| Weight decay | 0.01 |
| Batch | 8 reviews per step × 2 gradient-accumulation steps = effective batch 16 |
| Steps | 360 optimizer steps per epoch, 1,080 in total; warm-up ≈ 108 steps |
| Epochs | 3, evaluated after each; the checkpoint with the best validation macro-F1 is kept (best was epoch 2) |
| Max length | 512 tokens, dynamic padding per batch |
| Precision / memory | bf16 mixed precision, gradient checkpointing |
| Seed | 42 (Python, NumPy, PyTorch via `transformers.set_seed`) |
| Hardware | RTX 4060 Laptop GPU, 8 GB; i7-13th gen; 16 GB RAM |
| Cost | 13.0 min training, 5.6 GB peak GPU memory, ~20 s to score 1,437 reviews |

**Sanity checks before training (measured, `src/sanity_check.py`):** the untrained model's initial loss was 0.555
(close to ln 2 ≈ 0.693, as expected for a random head), and a 16-review batch was overfitted to loss ≈ 0 within 10
steps, confirming the pipeline can learn. Plain fp32 AdamW filled the GPU (7.9 GB) and stalled; 8-bit AdamW with
gradient checkpointing peaked at 5.6 GB.

**Validation behaviour of the final model** (`docs/figures/ettin400m_final/`): confusion matrix
[[192 TN, 18 FP], [26 FN, 1,201 TP]] with negative as class 0; ROC-AUC 0.985; negative-class PR-AUC 0.926;
Brier score 0.029; 97.6% of reviews get P > 0.95 or < 0.05.

---

## 4. Reproducibility (question b asks for this explicitly)

- **Fixed randomness:** seed 42 for weight initialisation, data order and the split.
- **Deterministic data split:** deduplication before splitting, so no review is in both train and validation.
- **Cross-validation:** 5-fold cross-validation of the same recipe on the training split (`src/cv_flag.py`,
  `outputs/cv/ettin400m_5fold`): out-of-fold accuracy 0.975, macro-F1 0.949 over 5,748 reviews. This agrees with the
  single validation split, so the result does not depend on one lucky split.
- **Run-to-run spread:** the same recipe trained twice scored 0.947 and 0.940 (GPU non-determinism; 98.3% of
  validation predictions agree). Eleven configurations on this split range from 0.925 to 0.947. We therefore treat
  differences under about 1 point as noise and use McNemar's exact test (paired, review by review) before claiming one
  setting beats another.
- **Pinned environment:** `requirements.txt` (Python 3.14, torch 2.11 + CUDA 12.6, transformers 5.19,
  bitsandbytes 0.50, scikit-learn 1.9, pandas 3.0).
- **Every run is logged:** settings and metrics in `outputs/runs/<run>/config.json` and `metrics.json`, per-epoch
  history in `history.json`, one row per run in `outputs/results.csv`.

---

## 5. Code and how to run (question b)

| File | Purpose |
|---|---|
| `src/data.py` | Loading, deduplication, the fixed 80/20 split, class weights |
| `src/train.py` | Fine-tuning (Hugging Face `Trainer`); writes run folder, appends to `outputs/results.csv`; `--save-model`, `--predict-test` |
| `src/sweep.py` | Runs a grid of `train.py` settings, one process each |
| `src/predict.py` | Saved model → `submission.csv` (+ `_with_probs.csv`) |
| `src/evaluate.py` | Confusion matrix, ROC, PR curve, calibration from a run's validation predictions |
| `src/cv_flag.py` | 5-fold out-of-fold predictions on the training split |
| `src/sanity_check.py` | Initial-loss and overfit checks |
| `src/cascade.py`, `src/cascade_embed.py` | Cascade experiments (section 7) |

```
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r requirements.txt
# Reproduce the submitted model and submission.csv (about 13 min on an 8 GB GPU)
.venv/Scripts/python src/train.py --run-name ettin400m_final --save-model --predict-test
# Evaluation figures
.venv/Scripts/python src/evaluate.py --run ettin400m_final
# Predict with a saved model
.venv/Scripts/python src/predict.py --model-dir outputs/runs/ettin400m_final/model --out submission.csv
```
Defaults in `train.py` are the final settings. On a GPU without bf16 (e.g. Colab T4) add `--fp16`.

---

## 6. How every parameter was chosen (question c)

**Principle:** one validation split for everything, macro-F1 as the selection metric, one change at a time, and a
paired significance test before calling a difference real.

| Parameter | Final value | How it was decided | Evidence |
|---|---|---|---|
| Model size | 400M | Largest Ettin encoder that fits 8 GB | 1B does not fit; 400M peaks at 5.6 GB |
| Learning rate | 2e-5 | Sweep {1e-5, 2e-5, 3e-5}; literature range 1e-5 to 5e-5 (Devlin et al. 2019; Mosbach et al. 2021) | 0.935 / **0.947** / 0.933; 3e-5 started unstably (0.901 after epoch 1) |
| Epochs | 3 + best-checkpoint selection | Small dataset, so overfitting risk; validation picks the epoch | A 4-epoch run peaked at epoch 2 and scored 0.936 |
| Warm-up | 10% of steps | The head starts random; warm-up stops early large updates from damaging pretrained weights (Mosbach et al. 2021) | Standard; not swept |
| Batch size | 16 (8 × 2 accumulation) | Memory limit of 8 GB; 16 is a standard fine-tuning batch | 360 updates per epoch |
| Max length | 512 | Token statistics on the training split (Ettin tokenizer): median 54, p95 181, p99 346, max 1,293 | Covers 99.67% of reviews whole; dynamic padding keeps short batches cheap |
| Class weights | None | Weighted loss is the usual fix for an 85/15 imbalance, so it was tested first | none **0.947**, neg-weight 2.0 0.938, 1.5 0.936, balanced 0.926; unweighted was better on both negative precision and recall |
| Weight decay | 0.01 | Standard AdamW regularisation for BERT-style fine-tuning | Not swept |
| Optimizer precision | 8-bit AdamW, bf16 | Memory only; 8-bit states cut optimizer memory ~4× with negligible accuracy loss (Dettmers et al. 2022) | fp32 AdamW did not fit |
| Selection metric | Macro-F1 | Always predicting positive already gets 85.4% accuracy, so accuracy hides failure on negatives | Majority class: accuracy 0.854, macro-F1 0.46 |
| Validation split | 80/20 | Compared with 90/10 | 0.935 vs 0.936, a tie; 80/20 gives a larger, more reliable validation set |
| Text preprocessing | Minimal (strip only) | Groupmate's whitespace normalisation tested on the same split | 0.943 vs 0.947, McNemar p = 0.83: no effect |
| Training data | Original | Synthetic negatives and label cleaning tested (details in `docs/report_draft_abc.md`) | +800 synthetic 0.925 (p = 0.013), label fixes 0.930 (p = 0.036): both worse |

**Learning-rate and epoch sweep** (no class weights; macro-F1 per epoch, best kept):

| Learning rate | Epochs | Acc | Macro-F1 | Neg-F1 | Macro-F1 by epoch | Time |
|---|---|---|---|---|---|---|
| 1e-5 | 3 | 0.967 | 0.935 | 0.889 | 0.932 / 0.935 / 0.935 | 12.6 min |
| **2e-5** | **3** | **0.973** | **0.947** | **0.909** | 0.926 / 0.942 / 0.947 | 11.9 min |
| 3e-5 | 3 | 0.967 | 0.933 | 0.886 | 0.901 / 0.925 / 0.933 | 12.5 min |
| 2e-5 | 4 | 0.969 | 0.936 | 0.890 | 0.929 / 0.936 / 0.933 / 0.932 | 16.7 min |

The gaps to the runner-up settings (1.1–1.4 points) are not significant (McNemar p = 0.11–0.29), so 2e-5 / 3 epochs is
the best measured setting rather than a proven winner. The final retrain with these settings scored 0.940; a realistic
expectation on new data is about 0.935–0.945.

**Class-weight sweep** (lr 2e-5, 3 epochs):

| Loss weights (negative / positive) | Acc | Macro-F1 | Neg-F1 | Neg precision / recall |
|---|---|---|---|---|
| **1.0 / 1.0 (none)** | **0.973** | **0.947** | **0.909** | 0.890 / 0.929 |
| 1.5 / 1.0 | 0.968 | 0.936 | 0.891 | 0.887 / 0.895 |
| 2.0 / 1.0 | 0.969 | 0.938 | 0.895 | 0.877 / 0.914 |
| 3.43 / 0.585 (balanced) | 0.962 | 0.926 | 0.874 | 0.858 / 0.891 |

Why weights did not help here: a pretrained encoder already separates the classes well, and the remaining errors are
mostly mixed reviews whose label follows the star rating. Up-weighting negatives pushes these borderline reviews
towards "negative" and costs more positives than it gains negatives.

---

## 7. Variants we built and tested (supporting material for b, c and e)

**Encoder vs decoder twin (same data, size and recipe).** `ettin_dec400m_lr2e-5`: Ettin decoder-400M fine-tuned with
the identical settings, classifying from the last token instead of mean pooling.

| | Acc | Macro-F1 | Neg-F1 |
|---|---|---|---|
| Ettin encoder-400M (final) | 0.969 | **0.940** | 0.897 |
| Ettin decoder-400M | 0.962 | 0.924 | 0.870 |

Encoder better (only encoder wrong on 9 reviews, only decoder wrong on 20; McNemar p = 0.061). The two share 35 of the
encoder's 44 errors. This matches the Ettin paper: bidirectional attention suits classification better than causal
(left-to-right) attention at equal size. A causal model's last token only sees the text before it; an encoder's every
token sees the whole review.

**Cascade: Ettin + Qwen3-8B for uncertain reviews (not submitted; used for question e).**
- Rule: if Ettin's confidence max(p, 1−p) < 0.99, the review goes to Qwen3-8B (4-bit, not trained), whose prompt holds
  the 8 most similar training reviews with their labels (retrieved with Qwen3-Embedding-0.6B).
- The 0.99 cut-off was chosen from {0.9, 0.99, 0.995, 0.999} on the 5-fold training predictions, never on validation.
- Validation: macro-F1 0.940 → 0.946; 59 of 1,437 reviews routed; 13 errors fixed, 8 introduced (McNemar p = 0.38, not
  significant). A plain vote over the same 8 reviews (0.937) and logistic regression on embeddings (0.930) did not
  help. Full details: `docs/ettin_gaps_and_hybrids.md`.
- Decision (2026-10-26): the submission stays the plain Ettin encoder; the cascade is used for the error analysis in
  (e) (`docs/report_draft_e.md`).

---

## 8. Results summary (validation, 1,437 reviews)

| Model | Acc | Macro-F1 | Neg-F1 |
|---|---|---|---|
| Majority class | 0.854 | 0.46 | 0.00 |
| TF-IDF (1–2-grams) + logistic regression | 0.911 | 0.766 | 0.582 |
| TF-IDF + logistic regression, balanced weights | 0.920 | 0.841 | 0.728 |
| SiEBERT zero-shot | 0.958 | 0.921 | 0.866 |
| Ettin decoder-400M fine-tuned | 0.962 | 0.924 | 0.870 |
| **Ettin encoder-400M fine-tuned (submitted)** | **0.969** | **0.940** | **0.897** |
| Ettin encoder-400M, best single run (same recipe) | 0.973 | 0.947 | 0.909 |
| Ettin + Qwen3-8B cascade (analysis only) | 0.973 | 0.946 | 0.907 |

Still to run: DeBERTa-v3-large (the industry-standard baseline) and the TF-IDF / BiLSTM baselines as committed
scripts (the TF-IDF numbers above come from exploratory runs).

---

## 9. References

- Weller et al. 2025, *Seq vs Seq: An Open Suite of Paired Encoders and Decoders* (Ettin), arXiv:2507.11412
- Warner et al. 2024, *ModernBERT*, arXiv:2412.13663
- Vaswani et al. 2017, *Attention Is All You Need*, arXiv:1706.03762
- Su et al. 2021, *RoFormer: Rotary Position Embedding*, arXiv:2104.09864
- Shazeer 2020, *GLU Variants Improve Transformer* (GeGLU), arXiv:2002.05202
- Devlin et al. 2019, *BERT*, arXiv:1810.04805
- He et al. 2021, *DeBERTaV3*, arXiv:2111.09543
- Mosbach et al. 2021, *On the Stability of Fine-tuning BERT*, arXiv:2006.04884
- Loshchilov & Hutter 2019, *Decoupled Weight Decay Regularization* (AdamW), arXiv:1711.05101
- Dettmers et al. 2022, *8-bit Optimizers via Block-wise Quantization*, arXiv:2110.02861
- Zhang et al. 2023, *Sentiment Analysis in the Era of Large Language Models: A Reality Check*, arXiv:2305.15005
- Bucher & Martini 2024, *Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification*, arXiv:2406.08660
- Hartmann et al. 2023, *More than a Feeling: Accuracy and Application of Sentiment Analysis* (SiEBERT), IJRM 40(1)
- McNemar 1947, *Note on the sampling error of the difference between correlated proportions or percentages*, Psychometrika 12
