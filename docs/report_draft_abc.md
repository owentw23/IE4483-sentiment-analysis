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
| Class weights | none | Balanced weights (3.43 / 0.585, from the 85/15 class ratio) were tested and lowered macro-F1 from 0.947 to 0.926; milder negative weights of 1.5 and 2.0 also lost (0.936, 0.938) |
| Weight decay | 0.01 | Standard AdamW regularisation for BERT-style fine-tuning |
| Selection metric | Macro-F1 | Always guessing "positive" already scores 85.4% accuracy, so accuracy hides failure on negatives. Macro-F1 weights both classes equally |
| Model size | 400M (fallback 150M) | 400M is the largest Ettin encoder that fits 8 GB with 8-bit AdamW and checkpointing. 1B does not fit |
| Optimizer precision | 8-bit AdamW, bf16 | Purely a memory constraint. 8-bit optimizer states cut memory ~4× with negligible accuracy loss |

[TBD: table of sweep results, learning rate → validation accuracy / macro-F1 / negative-F1]

### Training-data changes we tried and rejected

Most errors were on negative reviews (the minority class) and on reviews whose label seemed to contradict their text, so we tested two changes to the **training data**. The validation set was left untouched, so every run is scored on the same 1,437 reviews. Statistical significance uses McNemar's exact test, which compares two models review by review on that same set.

1. **Synthetic negative reviews.** With the instructor's approval, we added 800 LLM-written negative reviews (Claude, written from 16 reviewer personas after reading the training negatives). We aimed them at the hard cases: 330 mixed ("looks great, but…"), 107 implicit (sarcasm, understatement) and 363 plain. Formatting was matched to the real negatives (missing final punctuation, double spaces, length), and no synthetic review was closer than a TF-IDF cosine of 0.34 to any real review. We did not balance the classes 1:1, because balancing through class weights had already hurt.
2. **Label cleaning (confident learning).** We trained five Ettin models in 5-fold cross-validation on the training split, so every training review got a prediction from a model that had not seen it (Northcutt et al. 2021). 146 reviews disagreed with their label. We read each one: 32 were relabelled to match their text, 64 mixed or off-topic reviews were dropped, and 50 kept their label.

| Training data | Train size | Macro-F1 | Neg-F1 | Negatives missed | Positives missed | vs baseline |
|---|---|---|---|---|---|---|
| **Original (baseline)** | 5,748 | **0.947** | **0.909** | 15 | 24 | — |
| + 800 synthetic negatives | 6,548 | 0.925 | 0.872 | 30 | 23 | worse, p = 0.013 |
| Label cleaning | 5,684 | 0.930 | 0.882 | 20 | 31 | worse, p = 0.036 |
| Both | 6,484 | 0.931 | 0.882 | 19 | 32 | worse, p = 0.036 |

**Both changes made the model worse, so the final model uses the original training data.** Each configuration was trained once (seed 42), so the paired test covers which reviews were scored, not training randomness.

**Why synthetic data hurt: a style shortcut.** A simple TF-IDF + logistic regression detector told the synthetic negatives apart from real ones with 91.8% accuracy (5-fold). The cues were pure style: the LLM used contractions, present-tense stories and phrases like "two weeks later", while real customers wrote "this item was", "product", "Amazon" and "you". Half of the negative class now had a style that no real review has, so the model learned that real-customer writing leans positive (shortcut learning; Geirhos et al. 2020). Missed negatives doubled (15 → 30). The rise happened both for negatives with blunt words such as "junk" or "returned" (2 → 7 of 98) and without them (13 → 23 of 112). So the model became less sensitive to real negatives in general, rather than failing on one type. "Too blunt" is also ruled out: the synthetic reviews contain fewer blunt words than real negatives (19% vs 38%). This matches the finding that LLM-generated data helps least on subjective tasks such as sentiment (Li et al. 2023). Matching surface formatting was not enough; the voice also has to match.

**Why label cleaning hurt: the labels follow a rule, not random mistakes.** Most flagged reviews were not errors. They reflect how this dataset was labelled, apparently from star ratings: a mild complaint with a 4-star rating is labelled positive ("Nice sock, but… I will probably pass"), and praise followed by a deal-breaking flaw is labelled negative. The validation labels (and presumably the test labels) follow the same rule. Relabelling reviews to match their text taught the model a different rule from the one it is graded on, and dropping the mixed reviews removed the very examples that show where this dataset draws the line. The new errors were mostly mild-complaint reviews labelled positive (missed positives 24 → 31).

**Lessons:**
- Synthetic training data must be indistinguishable from real data. Otherwise its style becomes a feature the model can exploit.
- Label "noise" that follows the same pattern in training and test is part of the task, and cleaning it moves the model away from what is being measured. Confident learning is meant for random annotation errors.

(The label-cleaning result is also evidence for question (h): with inaccurate ratings, first check whether the errors are random or systematic before "correcting" them.)

### References
- Weller et al. 2025, *Seq vs Seq: An Open Suite of Paired Encoders and Decoders* (Ettin), arXiv:2507.11412
- Warner et al. 2024, *ModernBERT*, arXiv:2412.13663
- Devlin et al. 2019, *BERT*, arXiv:1810.04805
- Mosbach et al. 2021, *On the Stability of Fine-tuning BERT*, arXiv:2006.04884
- Loshchilov & Hutter 2019, *Decoupled Weight Decay Regularization* (AdamW), arXiv:1711.05101
- Dettmers et al. 2022, *8-bit Optimizers via Block-wise Quantization*, arXiv:2110.02861
- King & Zeng 2001, *Logistic Regression in Rare Events Data* (balanced class weights)
- Northcutt, Jiang & Chuang 2021, *Confident Learning: Estimating Uncertainty in Dataset Labels*, JAIR, arXiv:1911.00068
- Geirhos et al. 2020, *Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence, arXiv:2004.07780
- Li et al. 2023, *Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations*, EMNLP, arXiv:2310.07849
