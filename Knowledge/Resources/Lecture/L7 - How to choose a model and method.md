---
tags: [resources, lecture]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: sources below; project decisions from CLAUDE.md and docs/research_notes.md
---
# L7 — How to choose a model and method

Part of [[MOC - ML Training Lecture]]. Previous: [[L6 - Fine-tuning a pretrained transformer on one small GPU]].

Model selection isn't "pick the newest model". It's a funnel: **constraints → approach family → shortlist → baselines → validation decides**.

## 1. Step 1: write down the task, the metric and the constraints

| Question | This project |
|---|---|
| Task type | Binary text classification, fixed labels |
| Labelled data | 5.7k training reviews (after dedup and split), imbalanced 85/15 |
| Metric (fixed *before* experiments) | Macro-F1, plus accuracy and negative-F1 |
| Hardware | 8 GB laptop GPU; Colab T4 fallback |
| Time | A student project (hours of GPU, not weeks) |
| Rules | Train only on the training set; neural networks encouraged |
| Deliverables | Predictions plus a report that *explains and compares* |

## 2. Step 2: pick the approach family

| Situation | Usually best | Why |
|---|---|---|
| No labels | Zero-shot model or prompted LLM | Nothing to train on |
| Tens of labels | Few-shot (SetFit), or an LLM with examples | Too few to fine-tune a big model |
| **Thousands of labels, fixed classes** | **Fine-tune a pretrained encoder** | Purpose-built for understanding; cheap to run |
| Millions of labels, or CPU-only serving | Classical or small models can be enough | Data or cost dominate |
| Need generated explanations | Decoder LLM | Encoders don't generate |

We have thousands of labels and fixed classes, so **fine-tune an encoder**. The evidence: fine-tuned small and medium encoders consistently beat zero-shot or prompted LLMs on standard classification (Zhang et al. 2023; Bucher & Martini 2024). The Ettin paper shows encoders beat same-size decoders trained on identical data.

## 3. Step 3: shortlist with benchmarks, carefully

- **GLUE / SST-2** (movie-review sentiment) screen general understanding and sentiment ability. **MTEB** ranks *embedding* models, so it's relevant to the frozen-features arm, not to fine-tuning.
- A benchmark is a proxy. Check that it matches your setting (language, text length, fine-tuned vs zero-shot evaluation).
- Filter by constraints: the memory estimate from L6, licence, tokenizer, context length.

**Our shortlist (GLUE average, from the research notes):**

| Model | Params | GLUE | Fits 8 GB fine-tune? |
|---|---|---|---|
| DeBERTa-v3-large | 435M | 91.1 | With tricks |
| Ettin-encoder-400M | 400M | 90.8 | With tricks |
| ModernBERT-large | 396M | 90.4 | With tricks |
| Ettin-encoder-150M | 150M | 88.9 | Easily |
| Ettin-encoder-1B | 1B | 91.6 | **No** |

**An important honest observation.** On GLUE, DeBERTa-v3-large edges Ettin-400M (91.1 vs 90.8), and a ~0.3-point gap is within benchmark noise. So *Ettin isn't expected to clearly beat DeBERTa on accuracy*. The reasons for choosing Ettin are elsewhere:
- **Openness**: open training data and recipe.
- **A family of sizes**: 17M–1B, so scaling can be discussed.
- **The paired decoder twin**: an encoder-vs-decoder experiment where only the training objective differs.
- **Modern architecture**: RoPE, local/global attention, efficiency.

DeBERTa is therefore the right **"baseline to beat"**: the industry standard Ettin must match. Whichever wins on *our validation split* is the honest submission model. **The protocol decides, not the plan.**

## 4. Step 4: build a baseline ladder

Each rung answers a question and sets a bar for the next:

| Rung | Answers | Bar set |
|---|---|---|
| Majority class | What does "doing nothing" score? | 0.854 acc / 0.46 macro-F1 |
| TF-IDF + LogReg | How far do keywords get you? | 0.841 macro-F1 (weighted) |
| BiLSTM + GloVe | What do order and static meaning add? | to measure |
| Frozen embeddings + LogReg | What does pretrained knowledge add *without* training it? | to measure |
| SiEBERT zero-shot | What does generic sentiment knowledge give with no training on our data? | 0.921 |
| Fine-tuned Ettin / DeBERTa | What does adapting a pretrained model add? | projected 0.92–0.95 |

The **SiEBERT bar (0.921)** is a sharp test. A fine-tuned model that doesn't beat a zero-shot model hasn't justified its cost. The ladder is also the raw material for questions (f) and (g).

## 5. Step 5: let validation decide, and keep experiments honest

- **Same split, same seed and same metric for every arm.** Otherwise comparisons are meaningless (`CLAUDE.md` freezes this).
- **Ablation**: change exactly one thing and measure its effect. For example, class weights on vs off; synthetic negatives with vs without.
- **Fix the metric and the decision rule before looking at results.** Deciding afterwards which metric "matters" is a subtle form of cheating.
- **Respect noise**: under about 1 point = tie (L2).
- **Report negative results.** "Mamba-130M scored X, below the transformers" is valid and interesting evidence. Mamba has no published sentiment benchmarks, so any number is new.

## 6. Choosing *methods* (techniques), not just models

Use the same logic. A technique is worth including if it targets a **real, observed problem** at **acceptable cost**, and you can **measure** its effect.

| Technique | Problem it targets | Status |
|---|---|---|
| Class-weighted CE | 85/15 imbalance; the model under-learns negatives | **In.** Zero cost, measured +0.075 macro-F1 on TF-IDF |
| LLM synthetic negatives | Few negative examples (839) | **In, as an ablation.** Instructor permission pending |
| Others (R-Drop, model soups, multi-seed averaging, threshold tuning) | Stability, last-mile accuracy | **Dropped by the owner** to keep scope focused |

Rejected candidates (NeoBERT, LFM2.5, LLM2Vec, LoRA on 7B LLMs, chain-of-thought, and others) each failed one funnel step. They either targeted long inputs (ours are short), scored lower on the relevant benchmark, cost too much compute, or were shown to hurt binary sentiment. See `docs/research_notes.md`.

## 7. What would change the decision (contingencies)

- Ettin-400M runs out of memory even with the tricks → **Ettin-150M** (or Colab T4).
- Fine-tuning is unstable (validation stuck or collapsed) → lower η, check warm-up, re-check the pipeline (L5's sanity checks).
- DeBERTa wins on validation → submit DeBERTa and report it honestly. The comparison *is* the contribution.
- The instructor forbids synthetic data → drop Stage 4. Training-set-only is the brief's rule.

> [!question]- Check yourself 1: a classmate says "GPT-class LLMs are smarter, so prompt one instead of fine-tuning". What's your evidence-based reply?
> With thousands of labels and fixed classes, fine-tuned encoders usually beat zero-shot or prompted LLMs on classification (Zhang 2023; Bucher & Martini 2024). They're also far cheaper to run. LLM prompting shines when there are no labels, which is question (g)'s scenario, not ours.

> [!question]- Check yourself 2: why include a zero-shot SiEBERT result in a project about training models?
> It's a strong reference bar: a fine-tuned model must beat it to justify training. It also supplies evidence for question (g) about performing without labels.

> [!question]- Check yourself 3: Ettin-400M scores 0.951 macro-F1 and DeBERTa-v3-large scores 0.954 on validation. Which do you submit, and how do you write it up?
> They're statistically tied (difference under 1 point at about 210 negatives). Under the pre-registered rule, the higher validation macro-F1 wins, so submit DeBERTa. Report them as tied, and discuss efficiency and the encoder–decoder findings as Ettin's contributions. Don't claim either is "better".

## Sources and further study

- Zhang et al. 2023, *Sentiment Analysis in the Era of LLMs: A Reality Check*: https://arxiv.org/abs/2305.15005
- Bucher & Martini 2024, *Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification*: https://arxiv.org/abs/2406.08660
- Weller et al. 2025, *Seq vs Seq* (Ettin): https://arxiv.org/abs/2507.11412
- HF blog, *Introducing ModernBERT*: https://huggingface.co/blog/modernbert
- GLUE benchmark: https://gluebenchmark.com/ · MTEB leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- Andrew Ng, *Machine Learning Yearning* (baselines, dev sets, iteration strategy): https://info.deeplearning.ai/machine-learning-yearning-book
- Project research log: `docs/research_notes.md`
