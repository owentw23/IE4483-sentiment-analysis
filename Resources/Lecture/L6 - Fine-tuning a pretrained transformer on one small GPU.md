---
tags: [resources, lecture]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: sources below; hardware from CLAUDE.md
---
# L6 — Fine-tuning a pretrained transformer on one small GPU

Part of [[MOC - ML Training Lecture]]. Previous: [[L5 - Training in practice - hyperparameters and debugging]].

## 1. Transfer learning: why fine-tuning works with only 5.7k examples

Pretraining (masked-language modelling over about 2T tokens) teaches grammar, word meaning, negation, sarcasm cues and world knowledge. Fine-tuning only needs to teach "map that understanding to these two labels".

The cost is small learning rates (1e-5 to 5e-5). Large updates overwrite pretrained knowledge (**catastrophic forgetting**). Fine-tuning on small datasets is also known to be **unstable**: some runs fail or vary a lot across seeds. Mosbach et al. trace this to optimisation, not overfitting, and recommend small η with warm-up, Adam bias correction (on by default in AdamW) and enough training steps.

## 2. Will it fit? GPU memory anatomy

Training memory has four parts:
1. **Weights.**
2. **Gradients**: one per weight.
3. **Optimizer state**: AdamW keeps $m$ and $v$, two per weight.
4. **Activations**: forward-pass values stored for backprop. Remember the lecture: "store these forward values for the backward pass".

There's also roughly 0.5–1 GB of CUDA and framework overhead.

**Back-of-envelope for Ettin-400M (about 0.4 B parameters), plain fp32 AdamW:**

| Part | Bytes per parameter | Memory |
|---|---|---|
| Weights (fp32) | 4 | 1.6 GB |
| Gradients (fp32) | 4 | 1.6 GB |
| AdamW state, fp32 $m$ and $v$ | 8 | 3.2 GB |
| **Subtotal** | **16** | **6.4 GB** before activations and overhead |

On an 8 GB card that leaves almost nothing for activations, so it **doesn't fit**. (Hugging Face's guide quotes about 18 bytes/parameter for mixed-precision AdamW, because an extra half-precision weight copy is kept.)

**The four tricks the project uses:**

| Trick | What it does | Saves | Cost |
|---|---|---|---|
| **8-bit AdamW** (`bitsandbytes`) | Stores $m$ and $v$ in 8 bits with block-wise scaling | Optimizer state 3.2 GB → 0.8 GB | Negligible accuracy loss (Dettmers et al.) |
| **bf16 mixed precision** | Most matrix maths and activations in 16-bit bfloat (same range as fp32, less precision) | Roughly halves activation memory; much faster on tensor cores | Rarely, numerical issues; bf16 is safer than fp16 |
| **Gradient checkpointing** | Stores only some layer outputs and recomputes the rest during backward | Activation memory drops a lot | About 20–30% slower |
| **Small micro-batch + gradient accumulation** | Batch 8 forward/backward, accumulate twice, then update | Activation memory scales with micro-batch | More steps of wall-clock |

With 8-bit AdamW, the static part becomes about 4 + 4 + 2 = 10 bytes/parameter, about 4 GB. Checkpointing and bf16 keep activations for batch 8 × (up to 512 tokens) inside the rest. That's why `CLAUDE.md` says 400M fits *only* with these tricks, and 1B doesn't fit at all: 10 bytes × 1 B parameters = 10 GB before activations.

**Activation memory scales with batch × sequence length × hidden size × layers.** Dynamic padding (L3) helps a lot here: most reviews are about 54 tokens, not 512.

**Practical rule.** Estimate first, then *measure*: `torch.cuda.max_memory_allocated()` after a few steps. If you hit out-of-memory (OOM), halve the micro-batch and double the accumulation.

## 3. Software setup specifics for this laptop

- **GPU.** The RTX 5060 (Blackwell) needs a PyTorch build for **CUDA 12.8+**: `pip install torch --index-url https://download.pytorch.org/whl/cu128`. Older builds won't see the GPU.
- **Fallback.** Colab free T4 (16 GB, no bf16 tensor-core support, so use fp16 there; no FlashAttention 2).
- **Libraries.** `transformers` (models, `Trainer`), `datasets`, `scikit-learn` (splits, metrics, TF-IDF), `bitsandbytes` (8-bit optimizer), optionally `accelerate`.

## 4. The Hugging Face way (what the training script will look like)

```python
tok = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-400m")
model = AutoModelForSequenceClassification.from_pretrained("jhu-clsp/ettin-encoder-400m", num_labels=2)
args = TrainingArguments(
    learning_rate=2e-5, warmup_ratio=0.1, lr_scheduler_type="linear", weight_decay=0.01,
    per_device_train_batch_size=8, gradient_accumulation_steps=2, num_train_epochs=3,
    bf16=True, gradient_checkpointing=True, optim="adamw_bnb_8bit",
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="macro_f1", seed=42)
# Trainer subclass overrides compute_loss to apply CrossEntropyLoss(weight=[3.43, 0.585])
```

Every argument maps to a concept from L1, L2 or L5: η, schedule, decay, batch, accumulation, epochs, precision, memory, checkpoint selection, seed. Being able to justify every line *is* question (c).

## 5. Reproducibility (required by question b)

Sources of randomness:
- head initialisation
- data shuffling order
- dropout masks
- some non-deterministic GPU kernels

Controls:
- `transformers.set_seed(42)` seeds Python, NumPy and PyTorch.
- The split is fixed with `random_state=42`.
- Pin package versions (`requirements.txt`).
- Optionally `torch.use_deterministic_algorithms(True)`, which is slower and sometimes unsupported.

Even with seeds, results can differ across GPUs or library versions.

**Being honest about seed variance.** Studies such as Dodge et al. 2020 show that fine-tuning results vary noticeably across random seeds on small datasets. Averaging several seeds is the gold standard. The brief accepts any of: averaging over runs, cross-validation, or fixing the seed. The project owner chose **a fixed seed with a single run** (multiple seeds were dropped for compute). Pair that with the L2 rule: report differences under about 1 point as ties.

## 6. Where fine-tuning sits in question (f)

Fine-tuning is the most expensive format:
- **Memory**: all ~400M parameters need gradients and optimizer state.
- **Time**: every step runs backprop through 28 layers.
- **Energy**: GPU hours.
- **Inference**: a ~1.6 GB model (fp32), around a GPU-millisecond per review, vs microseconds for TF-IDF.

Frozen embeddings + logistic regression pay only the forward pass, once. Record peak memory, time per epoch and inference speed for each arm.

> [!question]- Check yourself 1: estimate fp32 AdamW static training memory for a 150M model. Does it fit 8 GB?
> 16 B × 0.15 B = 2.4 GB plus activations and overhead, so it fits comfortably without tricks. That's why 150M is the fallback.

> [!question]- Check yourself 2: you get CUDA out-of-memory at batch 8. What do you change first so the training maths stays the same?
> Micro-batch 4 with gradient accumulation 4. Effective batch stays 16, so η and the schedule are unchanged. Only activation memory drops.

> [!question]- Check yourself 3: why does fine-tuning need warm-up more than training a small net from scratch?
> The random head produces large, noisy gradients at the start, and Adam's moment estimates aren't settled yet. Warm-up keeps those first updates from wrecking the pretrained encoder weights.

## Sources and further study

- Hugging Face LLM Course, ch. 3 (fine-tuning with `Trainer`): https://huggingface.co/learn/llm-course/en/chapter3/3
- HF docs, efficient training on one GPU (bf16, accumulation, checkpointing, 8-bit optimizers): https://huggingface.co/docs/transformers/en/perf_train_gpu_one
- HF docs, model training anatomy (where memory goes): https://huggingface.co/docs/transformers/en/model_memory_anatomy
- EleutherAI, *Transformer Math 101*: https://blog.eleuther.ai/transformer-math/
- Mosbach et al. 2021, *On the Stability of Fine-tuning BERT*: https://arxiv.org/abs/2006.04884
- Dodge et al. 2020, *Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping*: https://arxiv.org/abs/2002.06305
- Dettmers et al. 2022, *8-bit Optimizers via Block-wise Quantization*: https://arxiv.org/abs/2110.02861
- PyTorch, reproducibility notes: https://pytorch.org/docs/stable/notes/randomness.html
