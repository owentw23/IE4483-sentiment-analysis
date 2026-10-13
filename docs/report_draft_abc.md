# Report draft — Questions (a), (b), (c) — Ettin encoder

Items marked **[TBD]** are filled in after the training runs.

## (a) Feature format and preprocessing

**Choice of feature format**
- **Contextual embeddings from a pretrained language model, fine-tuned end-to-end** (`jhu-clsp/ettin-encoder-400m`).
- The raw text is converted into **subword token IDs + an attention mask** (pipeline below). The encoder then turns these into one 1024-dimensional contextual vector per token.
- Because the encoder is fine-tuned, these embeddings keep adapting to our task during training. The network that produces the features and the classifier are trained together (architecture in b).

**Why contextual embeddings: comparison of the three feature formats in the brief**

| | TF-IDF (bag-of-words) | Static word embeddings (GloVe) | Contextual embeddings (Ettin) |
|---|---|---|---|
| Word representation | Weighted count in one column | Fixed 300-d vector per word | 1024-d vector computed from the whole review |
| Captures meaning (synonyms) | No | Yes | Yes |
| Depends on context | No | No ("good" identical everywhere) | Yes ("good" differs in "not good") |
| Word order / negation | Lost | Only if the model reads in sequence (e.g. LSTM) | Built in via self-attention |
| Representation | Sparse, 14,114-d | Dense, 300-d per word | Dense, 1024-d per token |
| Source | Counted on our training set | Lookup table pretrained once | Forward pass through the pretrained encoder |

- Negation and contrast ("great *but* fell apart") decide sentiment in many reviews, and only contextual embeddings represent them directly.
- Contextual embeddings carry knowledge from ~2T tokens of pretraining, which matters with only 5.7k training reviews.
- Measured on our validation split:
  - TF-IDF + LogReg (class-weighted): 0.920 accuracy, 0.841 macro-F1
  - A pretrained transformer used zero-shot (SiEBERT): 0.958 accuracy, 0.921 macro-F1
- The resource-vs-accuracy trade-off between these formats is discussed in (f).

**Preprocessing pipeline (raw text → model input)**
1. **Load** `train.json` / `test.json` with pandas (`reviews` → text, `sentiments` → integer label 0/1).
2. **Deduplicate** 216 exact duplicate reviews in the training set, keeping the first (7,401 → 7,185). No duplicate had conflicting labels. This stops the same review appearing in both the train and validation splits.
3. **Split** into train (5,748) and validation (1,437) with an 80/20 **stratified** split (`random_state=42`), preserving the 85/15 class ratio in both.
   - Train: 839 negative / 4,909 positive.
4. **Minimal text cleaning, on purpose:**
   - No lowercasing: the tokenizer is case-sensitive, and capitals carry sentiment ("TERRIBLE").
   - No stop-word removal: "not", "but" and "never" decide sentiment.
   - No stemming or lemmatisation: the subword tokenizer already handles word variants.
   - Punctuation and emoji kept: "!!" is its own token.
   - Only leading/trailing whitespace is stripped.
5. **Tokenise** with Ettin's pretrained BPE tokenizer (vocabulary 50,368).
   - Example: `"Not good, returned it!!"` → `[CLS] Not Ġgood , Ġreturned Ġit !! [SEP]` (Ġ marks a leading space).
   - `[CLS]` and `[SEP]` are added automatically.
6. **Truncate** to `max_length = 512` tokens.
   - Token-length statistics on the training data: median 54, 95th percentile 181, 99th percentile 346, max 1,293.
   - 512 covers 99.67% of reviews whole. Only 0.33% are cut.
7. **Pad dynamically** per batch to the longest review in that batch, rather than always to 512. This saves compute because most reviews are short. An **attention mask** (1 = real token, 0 = padding) tells the model to ignore padding.
8. **Labels** become integer tensors, 0 = negative and 1 = positive.
9. **Test set** gets exactly the same tokenizer and settings. The tokenizer is pretrained and fixed, so nothing is fitted on test data and there's no leakage.
10. **Class weights** are computed from the training split only (see b).

## (b) Model

**Model: Ettin-encoder-400M + classification head (fully fine-tuned)**
- Encoder-only transformer from the Ettin suite (Weller et al., 2025, ICLR 2026). It uses the ModernBERT architecture and was pretrained with masked language modelling on ~2T open tokens.
- About 396M parameters, all fine-tuned.

**Architecture figure (to draw)**
```
Review text
  → Tokenizer (BPE, 50,368 vocab)          → input_ids, attention_mask   [B, L]
  → Token embedding                         →                             [B, L, 1024]
  → 28 × Transformer layer                  →                             [B, L, 1024]
       (multi-head self-attention, 16 heads, RoPE positions,
        local 128-token window; every 3rd layer global;
        GeGLU feed-forward 1024→2624→1024; pre-LayerNorm; residuals)
  → Masked mean pooling over real tokens    →                             [B, 1024]
  → Head: Dense 1024→1024 + GELU + LayerNorm →                            [B, 1024]
  → Linear 1024→2                           → logits                      [B, 2]
  → Softmax                                 → P(negative), P(positive)    [B, 2]
```
- **Input:** a batch of `B` reviews, each up to `L ≤ 512` token IDs plus an attention mask.
- **Output:** 2 logits per review. Softmax gives the class probabilities, and the predicted label is the larger one.
- **Pooling:** masked mean of all token vectors (Ettin's default `classifier_pooling = "mean"`), so every word contributes, not just `[CLS]`.

**Loss function**
- **Class-weighted cross-entropy:** `L = −w_y · log p_y`
- Balanced weights from the training split, `w_c = N / (2·N_c)`: negative = **3.43**, positive = **0.585**.
- So misclassifying a negative costs ~5.9× more, which counteracts the 85/15 imbalance.
- Evidence: on the TF-IDF baseline, class weights raised macro-F1 from 0.766 to 0.841.

**Training strategy**
- Full fine-tuning: all encoder weights plus the randomly initialised head are updated.
- **Optimizer:** AdamW, 8-bit (`bitsandbytes`) to fit 8 GB of GPU memory, with weight decay 0.01.
- **Learning-rate schedule:** linear warm-up over the first 10% of steps, then linear decay to 0.
- **Batch:** 8 per step × 2 gradient-accumulation steps = effective batch 16.
- **Precision:** bf16 mixed precision, plus gradient checkpointing to save memory.
- **Epochs:** up to 3. Evaluate on validation after each epoch and keep the checkpoint with the **best validation macro-F1**, which acts as early stopping.
- **Final model:** the best checkpoint predicts `test.json` → `submission.csv`.
- **Hardware:** RTX 4060 Laptop GPU, 8 GB. [TBD: training time from the final run]

**Reproducibility**
- Fixed seed 42 for Python, NumPy and PyTorch (`transformers.set_seed(42)`), plus a deterministic split (`random_state=42`).
- Package versions pinned in `requirements.txt`.
- Single run per configuration. Differences under ~1 point are within validation noise (±1 point at n = 1,437) and are not claimed as significant.

**Code and run instructions**
- Code: `src/data.py` (loading, deduplication, split, class weights), `src/train.py` (fine-tuning; Hugging Face `Trainer` with a class-weighted loss), `src/sweep.py` (hyperparameter grid), `src/predict.py` (test predictions → `submission.csv`), `src/sanity_check.py` (initial-loss and overfit checks).
- Environment: Python 3.14, PyTorch 2.11 (CUDA 12.6), transformers 5.19, bitsandbytes 0.50 (`requirements.txt`); RTX 4060 Laptop GPU, 8 GB.
- Run:
  ```
  python -m venv --system-site-packages .venv
  .venv/Scripts/python -m pip install -r requirements.txt
  .venv/Scripts/python src/train.py --run-name FINAL --lr <chosen> --save-model --predict-test
  ```
- Sanity checks before training (measured): initial weighted loss 0.555 (≈ ln 2 = 0.693, as expected for an untrained head); a 16-review batch was overfitted to loss ≈ 0 within 10 steps, confirming the pipeline learns.
- Memory (measured): plain fp32 AdamW filled the 8 GB GPU (7.9 GB) and stalled; 8-bit AdamW with gradient checkpointing peaked at about 6 GB.

## (c) How parameters were chosen

| Parameter | Value | How and why |
|---|---|---|
| Learning rate | [TBD: best of {1e-5, 2e-5, 3e-5}] | Large pretrained encoders forget their pretraining or diverge at high learning rates. Fine-tuning studies (Devlin et al. 2019, Mosbach et al. 2021) recommend 1e-5 to 5e-5. A 3-value sweep, picking the best validation macro-F1 |
| Warm-up | 10% of steps | The head starts random, so large early updates would disrupt the pretrained weights. Warm-up stabilises the start (Mosbach et al. 2021) |
| Epochs | ≤ 3 + best-checkpoint selection | Small dataset (5.7k), so more epochs risk overfitting. Validation macro-F1 decides which epoch to keep |
| Batch size | 8 × 2 accumulation = 16 | Limited by the 8 GB GPU. 16 is a standard fine-tuning batch size and gives ~360 updates per epoch on 5.7k reviews |
| Max length | 512 | From token statistics: covers 99.67% of reviews whole. Dynamic padding keeps the cost low |
| Class weights | 3.43 / 0.585 | Computed from the class ratio, not tuned. Fixes the 85/15 imbalance |
| Weight decay | 0.01 | Standard AdamW regularisation for BERT-style fine-tuning |
| Selection metric | Macro-F1 | Always guessing "positive" already scores 85.4% accuracy, so accuracy hides failure on negatives. Macro-F1 weights both classes equally |
| Model size | 400M (fallback 150M) | 400M is the largest Ettin encoder that fits 8 GB with 8-bit AdamW and checkpointing. 1B does not fit |
| Optimizer precision | 8-bit AdamW, bf16 | Purely a memory constraint. 8-bit optimizer states cut memory ~4× with negligible accuracy loss |

[TBD: table of sweep results, learning rate → validation accuracy / macro-F1 / negative-F1]

### References
- Weller et al. 2025, *Seq vs Seq: An Open Suite of Paired Encoders and Decoders* (Ettin), arXiv:2507.11412
- Warner et al. 2024, *ModernBERT*, arXiv:2412.13663
- Devlin et al. 2019, *BERT*, arXiv:1810.04805
- Mosbach et al. 2021, *On the Stability of Fine-tuning BERT*, arXiv:2006.04884
- Loshchilov & Hutter 2019, *Decoupled Weight Decay Regularization* (AdamW), arXiv:1711.05101
- Dettmers et al. 2022, *8-bit Optimizers via Block-wise Quantization*, arXiv:2110.02861
- King & Zeng 2001, *Logistic Regression in Rare Events Data* (balanced class weights)
