# IE4483 Project 1 — Sentiment Analysis (project context for Claude Code)

## Goal
Build a learning-based binary sentiment classifier for product reviews (0 = negative, 1 = positive),
predict `test.json`, submit `submission.csv`, and write a report answering questions (a)–(h)
in `instruction2.png`. The owner wants the project to stand out vs the industry standard
(fine-tuned RoBERTa / DeBERTa) at student scale.

## Data facts (verified)
- `train.json`: 7,401 rows, columns `reviews`, `sentiments`. 6,319 positive / 1,082 negative (85/15).
- `test.json`: 1,851 rows, `reviews` only (no labels).
- 216 duplicate texts in train (no conflicting labels); 87 train texts also appear in test.
- Median review ~42 words, p95 ~145 words; only 25 train / 3 test reviews exceed ~512 tokens.
- Labels look star-rating derived: many errors are mixed reviews ("looks great but fell apart"),
  some look mislabelled → expect ~2–3% irreducible error.
- The brief says "csv" but files are JSON; submission.csv column format is unspecified — confirm with instructor.
- Brief: training must use only the training set.

## Validation protocol (keep identical across all models)
```python
df = pd.read_json("train.json").drop_duplicates("reviews").reset_index(drop=True)
tr, va = train_test_split(df, test_size=0.2, stratify=df.sentiments, random_state=42)
```
Single fixed seed (42). Report accuracy, **macro-F1**, and negative-class F1.

## Decisions made — do not re-propose dropped items
- **Main model:** `jhu-clsp/ettin-encoder-400m` (fallback `jhu-clsp/ettin-encoder-150m`).
  Same architecture/size/speed as ModernBERT-large (`model_type: modernbert`); Ettin chosen for
  open data, more sizes, and the paired decoder twin.
- **Baseline to beat:** `microsoft/deberta-v3-large`.
- **Other arms:** TF-IDF + LogReg; BiLSTM + GloVe; SiEBERT zero-shot
  (`siebert/sentiment-roberta-large-english`); optional Ettin decoder twin
  (`jhu-clsp/ettin-decoder-400m`) for encoder-vs-decoder; `state-spaces/mamba-130m-hf` (modern RNN,
  needs CUDA kernels → Linux/WSL2/Colab); `Qwen/Qwen3-Embedding-0.6B` + LogReg (frozen features);
  GLiClass (`knowledgator/gliclass-modern-base-v2.0-init`) and `vllm-sr/Decision-2.0-Kai-0.6B`
  zero-shot for report question (g).
- **Techniques: class-weighted loss + LLM-generated synthetic negatives (with/without ablation) only.**
  Dropped by the owner: R-Drop, model soups/weight averaging, multiple seeds, threshold tuning.
- Explicitly rejected: chain-of-thought prompting, LLM2Vec, xLSTM/Jamba/Zamba, LoRA on 7B LLMs,
  LFM2.5 encoders (worse SST-2 than ModernBERT-large), NeoBERT, mmBERT.
- Class weights (balanced, N/(K·N_c)) on full train: negative ≈ 3.42, positive ≈ 0.59 — recompute after dedup/split.

## Results so far (validation split above, 1,437 reviews)
| Model | Acc | Macro-F1 | Neg-F1 |
|---|---|---|---|
| Majority class | 0.854 | 0.46 | 0.00 |
| TF-IDF (1–2 gram) + LogReg | 0.911 | 0.766 | 0.582 |
| TF-IDF + LogReg, class_weight=balanced | 0.920 | 0.841 | 0.728 |
| SiEBERT zero-shot | 0.958 | 0.921 | 0.866 |
Projected Ettin-400M fine-tuned + class weights: ~96–97.5% acc, ~0.92–0.95 macro-F1.

## Hardware
- **Windows laptop (primary for training):** RTX 5060 laptop, **8 GB VRAM**, i7-13th gen, 16 GB RAM.
  - Needs PyTorch CUDA 12.8+: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
  - Fits easily: TF-IDF, BiLSTM, embeddings, zero-shot, Ettin-150M, Mamba-130M (WSL2).
  - Ettin-400M / DeBERTa-v3-large: need 8-bit AdamW (`bitsandbytes`), gradient checkpointing,
    batch 8 + gradient accumulation, bf16. Ettin-1B does not fit.
- **Colab free T4 (16 GB):** fallback for 400M-class models. No FlashAttention 2 on T4.
- **MacBook M4 16 GB:** quick checks only (MPS).

## Next steps
1. Shared data/split module + TF-IDF and BiLSTM baselines as scripts.
2. Ettin-400M (or 150M) fine-tune with class-weighted loss → validation metrics → `submission.csv`.
3. DeBERTa-v3-large baseline, then optional arms (Ettin decoder, Mamba, embeddings, zero-shot).
4. Synthetic negatives ablation (check with instructor that LLM-generated data is allowed).
5. Report: answer (a)–(h); use SiEBERT error samples for (e); cleanlab / co-teaching /
   label smoothing as the three approaches for (h); zero-shot + domain-adaptive pretraining + SetFit for (g).

See `docs/research_notes.md` for how every model/technique works and the source papers.
Report drafts: `docs/report_draft_abc.md` (questions a–c, Ettin; [TBD] items filled after training runs).
Verified for the report: Ettin-400m uses `classifier_pooling: mean`; token lengths (Ettin tokenizer, deduped train)
median 54 / p95 181 / p99 346 / max 1,293 → max_length 512 covers 99.67%; train split 839 neg / 4,909 pos →
class weights neg 3.426, pos 0.585.
