# Research Notes — Models, Techniques, Sources

Condensed from the research phase (literature via Perplexity + web, Oct 2026). Each entry: how it
works, why it is in the project, and sources for building it.

## Shared ideas
1. **Tokenisation** — text → units (words for TF-IDF/GloVe, sub-words for transformers).
2. **Embedding** — each unit → learned vector.
3. **Head + softmax** — final linear layer → 2 scores → probabilities.
4. **Cross-entropy** — loss = −log p(correct label); training minimises it via gradient descent.

---

## Models in the plan

### 1. TF-IDF + Logistic Regression (classical baseline)
- TF = count of word in review; IDF = log(N / reviews containing word) → rare words weigh more.
- Each review → sparse 14k-dim vector; LogReg learns one weight per word: p = sigmoid(w·x + b).
- Ignores word order (bigrams partly fix "not good").
- Sources: [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), course `tfidf_features_example.ipynb`.

### 2. BiLSTM + GloVe (classic RNN)
- GloVe: pre-trained 300-d word vectors from co-occurrence statistics.
- LSTM reads word by word with hidden state h and cell c; forget/input/output gates control memory;
  additive cell update avoids vanishing gradients. Bidirectional = left→right + right→left, concatenated.
- Max-pool over time → dropout → linear → softmax.
- Sources: [Hochreiter & Schmidhuber 1997](https://www.bioinf.jku.at/publications/older/2604.pdf), [GloVe paper](https://nlp.stanford.edu/pubs/glove.pdf), [GloVe vectors](https://nlp.stanford.edu/projects/glove/), [nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html).

### 3. Transformer encoders: RoBERTa, DeBERTa-v3, SiEBERT (industry standard)
- Self-attention: each token forms Query/Key/Value; new vector = softmax(Q·K/√d)-weighted sum of Values.
  Multi-head, stacked 12–24 layers, bidirectional (sees whole review).
- Pretrained with masked language modelling; fine-tune by putting a linear head on the first token
  and training all weights (lr ~2e-5, 2–4 epochs).
- RoBERTa = BERT with 10× data, dynamic masking, no next-sentence prediction.
- DeBERTa-v3 = disentangled content/position attention + ELECTRA-style replaced-token detection.
- SiEBERT = RoBERTa-large already fine-tuned on 15 sentiment datasets; used zero-shot as reference.
- Sources: [Transformer](https://arxiv.org/abs/1706.03762), [BERT](https://arxiv.org/abs/1810.04805), [RoBERTa](https://arxiv.org/abs/1907.11692), [DeBERTaV3](https://arxiv.org/abs/2111.09543), [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large), [SiEBERT](https://huggingface.co/siebert/sentiment-roberta-large-english) (Hartmann et al. 2023, IJRM 40(1)), [HF sequence-classification tutorial](https://huggingface.co/docs/transformers/tasks/sequence_classification).
- Evidence fine-tuned encoders beat prompted LLMs: [Bucher & Martini 2024](https://arxiv.org/abs/2406.08660) ([code](https://github.com/mnbucher/text-cls-llms)), [Zhang et al. 2023 Reality Check](https://arxiv.org/abs/2305.15005).

### 4. ModernBERT
- BERT rebuilt: 2T training tokens (incl. code), RoPE rotary positions (8k context), alternating
  global/local (128-token) attention every 3rd layer, unpadding + FlashAttention, GeGLU, pre-norm.
- ~2× faster / 5–7× less memory than DeBERTa-v3 — but only with FlashAttention 2 (Ampere+ GPU) and long inputs.
- GLUE: base 88.4, large 90.4 (DeBERTa-v3-large 91.1).
- Sources: [paper](https://arxiv.org/abs/2412.13663), [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large).

### 5. Ettin encoder / decoder (main model + controlled experiment)
- Twin models per size (17M–1B): identical open data (~2T tokens), ModernBERT architecture, recipe;
  only difference is objective — encoder: masked LM, bidirectional; decoder: next-token, causal.
- Not "BERT + ChatGPT": two separate models. Decoder is a small GPT-style LM, not a chatbot.
- Configs identical to ModernBERT at matching sizes (150M ≈ ModernBERT-base 149.7M; 400M ≈ ModernBERT-large 395.9M) → same speed.
- vs BERT: ~600× data, no NSP, higher mask rate, 3-phase schedule (1.7T pretrain → 250B context extension → decay on high-quality data), RoPE, local/global attention, GeGLU.
- GLUE: 150M 88.9 (vs ModernBERT-base 88.4); 400M 90.8 (vs ModernBERT-large 90.4); 1B 91.6, SST-2 97.1.
- Encoders beat same-size decoders on classification; converting a decoder to an encoder with 50B more tokens does not close the gap.
- Decoder classification: use last non-padding token hidden state → linear.
- Sources: [paper (ICLR 2026)](https://arxiv.org/abs/2507.11412), [ettin-encoder-400m](https://huggingface.co/jhu-clsp/ettin-encoder-400m), [ettin-decoder-400m](https://huggingface.co/jhu-clsp/ettin-decoder-400m).

### 6. Mamba-130M (modern RNN / state-space model)
- Recurrence h_t = A·h_{t−1} + B·x_t, y_t = C·h_t. "Selective": B, C and step Δ depend on the input,
  so the model chooses per token what to absorb or ignore (gating). Parallel scan → fast GPU training.
  No attention; linear in length. Classify from last token's output.
- No published standard-sentiment benchmarks — results are new evidence. Expect below transformers.
- Needs CUDA kernels (`mamba-ssm`, `causal-conv1d`) → Linux/WSL2/Colab. No ready classification class: add own head.
- Sources: [Mamba paper](https://arxiv.org/abs/2312.00752), [mamba-130m-hf](https://huggingface.co/state-spaces/mamba-130m-hf), [HF docs](https://huggingface.co/docs/transformers/en/model_doc/mamba), [code](https://github.com/state-spaces/mamba), [Mamba-2 blog](https://goombalab.github.io/blog/2024/mamba2-part1-model/).

### 7. Qwen3-Embedding-0.6B + LogReg (frozen LLM features)
- Decoder LLM trained contrastively so similar texts get similar vectors; last-token 1024-d embedding;
  optional instruction prefix. Frozen — extract once, fit LogReg in seconds.
- MTEB English v2 classification 85.76.
- Sources: [paper](https://arxiv.org/abs/2506.05176), [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

### 8. Zero-shot: GLiClass, Decision-2.0 (report question g)
- GLiClass: ModernBERT encoder; labels + text in one sequence; label tokens scored against text in one pass.
- Decision-2.0 (vLLM Semantic Router): Qwen3-based decision models (0.6B–27B) that output a probability
  per option in one forward pass; built for routing, no published sentiment benchmark; "Jev" = their own benchmarks (JevArena).
- Sources: [GLiClass paper](https://arxiv.org/abs/2508.07662), [gliclass-modern-base-v2.0-init](https://huggingface.co/knowledgator/gliclass-modern-base-v2.0-init) (`pip install gliclass`), [BTZSC 2026 zero-shot benchmark](https://arxiv.org/abs/2603.11991), [Decision-2.0-Kai-0.6B](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B).

---

## Techniques in the plan

### A. Class weights
- loss = −w_y · log p(correct); balanced w_c = N / (K × N_c) → negative ≈ 3.42, positive ≈ 0.59
  (a negative error costs ~5.8× a positive error).
- Measured: TF-IDF macro-F1 0.766 → 0.841 with balanced weights.
- Build: sklearn `class_weight="balanced"`; PyTorch `nn.CrossEntropyLoss(weight=...)`; HF Trainer → override `compute_loss`.
- Sources: [King & Zeng 2001](https://gking.harvard.edu/files/0s.pdf), [compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html), [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

### B. Synthetic negatives (LLM augmentation)
1. Prompt an LLM with 3–5 real negatives + varied spec (product category, complaint type, length, tone incl. mixed).
2. Generate ~1–2k negatives.
3. Filter: dedupe, drop near-copies of real train/test reviews, hand-check 50.
4. Add to training split only; validation/test stay real.
5. Ablation: with vs without.
- Findings: most effective in low-resource settings; mix with real data; more synthetic ≠ better; watch LLM-style artefacts; less effective for more subjective tasks.
- Check with instructor whether allowed.
- Sources: [Li et al. 2024](https://arxiv.org/abs/2407.12813), [Li et al. EMNLP 2023](https://arxiv.org/abs/2310.07849).

---

## Report-only concepts
| Concept | How it works | Question | Source |
|---|---|---|---|
| Confident learning / cleanlab | Out-of-sample probabilities via CV; per-class confidence thresholds; count labelled-i-but-confidently-j → flag label errors | (e), (h) | [arXiv 1911.00068](https://arxiv.org/abs/1911.00068), [cleanlab](https://github.com/cleanlab/cleanlab) |
| Co-teaching | Two nets; each trains its peer on its small-loss (likely clean) examples | (h) | [arXiv 1804.06872](https://arxiv.org/abs/1804.06872) |
| Label smoothing | Targets 0.9/0.1 instead of 1/0 → less damage from wrong labels | (h) | [arXiv 1906.02629](https://arxiv.org/abs/1906.02629) |
| BERT label-noise robustness | Effect of flipped labels on BERT and which fixes help | (h) | [arXiv 2204.09371](https://arxiv.org/abs/2204.09371) |
| Domain-adaptive pretraining | Continue masked-LM on unlabelled hotel reviews before fine-tuning | (g) | [arXiv 2004.10964](https://arxiv.org/abs/2004.10964) |
| SetFit | Few labels → contrastive pairs → fine-tune sentence encoder → LogReg | (g) | [arXiv 2209.11055](https://arxiv.org/abs/2209.11055), [setfit](https://github.com/huggingface/setfit) |

## Researched and rejected
| Candidate | Reason |
|---|---|
| NeoBERT (250M) | Gains on long inputs; reviews are short; below Ettin-400M |
| LFM2.5-Encoder-350M (Jul 2026) | SST-2 95.1 vs ModernBERT-large 96.1; wins are multilingual / CPU long-context |
| mmBERT | English GLUE below ModernBERT |
| OptiBERT | No HF checkpoint; authors say top models not statistically distinguishable |
| Julia-1 (Sep 2026) | 100-example pilots only, no sentiment results, no paper |
| LoRA on decoder LLMs | Ettin decoder answers encoder-vs-decoder more cleanly and cheaply |
| LLM2Vec | ~4.7 A100-hours adaptation |
| RWKV-7 / xLSTM / minGRU / Jamba / Zamba | Mamba covers modern RNN; others too large or need extra libraries |
| Chain-of-thought | Hurts binary sentiment (GPT-4o macro-F1 0.895 direct vs 0.841 CoT, [arXiv 2506.04574](https://arxiv.org/abs/2506.04574)) |
| Ettin-1B | Doesn't fit 8 GB laptop or free T4 for full fine-tuning |

## Industry/benchmark context
- Fine-tuned small encoders beat zero-shot LLMs on standard sentiment ([Zhang 2023](https://arxiv.org/abs/2305.15005), [Bucher & Martini 2024](https://arxiv.org/abs/2406.08660)).
- Decision models / zero-shot classifiers suit changing labels or no labels; fine-tuned encoders suit stable labels + thousands of examples.
- GPU options: Kaggle (T4/P100, ~30 h/week), Colab free (T4), Vast.ai/RunPod RTX 4090 ~$0.13–0.50/hr.
