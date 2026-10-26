---
tags: [resources, lecture, hub]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: claude-lecture; sources listed per module
---
# Lecture: from backprop to a trained sentiment classifier

**Who this is for.** You know transformers, attention, LLM basics and the idea of backprop. You haven't trained or selected a model yet. Your course is up to *EE4483 – Neural Networks and Backpropagation*, which you haven't read. Module 1 covers that lecture and then fills in what it leaves out.

**Goal.** By the end you should be able to:
1. Explain every design choice in this project.
2. Answer report questions (a)–(h) from understanding rather than copying.
3. Make and defend model and method decisions yourself.

## Modules (read in order)

| # | Module | Answers mostly |
|---|---|---|
| 1 | [[L1 - From backprop to a real training run]] | Background for (b), (c) |
| 2 | [[L2 - Data splits, metrics and class imbalance]] | (b), (c), (d) |
| 3 | [[L3 - Turning text into features]] | (a), (f) |
| 4 | [[L4 - Classifier architectures]] | (b) |
| 5 | [[L5 - Training in practice - hyperparameters and debugging]] | (b), (c) |
| 6 | [[L6 - Fine-tuning a pretrained transformer on one small GPU]] | (b), (c), (f) |
| 7 | [[L7 - How to choose a model and method]] | The project's decisions; (b), (f) |
| 8 | [[L8 - Error analysis, domain shift and noisy labels]] | (e), (g), (h) |

Each module ends with **check-yourself questions**. Their answers are in folded callouts, so click to expand them.

## How the report questions map to concepts

| Q | What the marker wants | Core concepts |
|---|---|---|
| (a) 10% | The feature format, and every step from raw text to model input | Tokenisation, TF-IDF vs embeddings, splits before fitting, truncation and padding (L2, L3) |
| (b) 20% | An architecture figure with input/output sizes, the loss, the training strategy, code and how to run it, and reproducibility | Heads and pooling, cross-entropy, optimizer, schedule, checkpoint selection, seeds (L1, L4–L6) |
| (c) 10% | *How* each hyperparameter was chosen and *why* | Learning-rate sweeps, literature defaults, constraints set by data and hardware (L5, L6) |
| (d) 20% | `submission.csv` with test predictions | Inference, matching the instructor's format (L2) |
| (e) 10% | 1–2 correct and incorrect cases and what they show | Error analysis and error taxonomy (L8) |
| (f) 10% | How the feature format changes resource use and accuracy | Parameters, memory, time vs accuracy across TF-IDF, GloVe, frozen and fine-tuned embeddings (L3, L6) |
| (g) 10% | New domain (hotels) with **no labels**: how it performs and how to adapt | Domain shift, zero-shot, domain-adaptive pretraining, few-shot (L8) |
| (h) 10% | Labels exist but are **noisy**: how it performs, plus three fixes | Memorisation, confident learning, co-teaching, label smoothing (L8) |

## The project in one paragraph

You have 7,401 labelled product reviews, 85% of them positive, and 1,851 unlabelled test reviews. The plan:
- Remove duplicate reviews and hold out a stratified 20% as validation.
- Build a ladder of models, each stronger and more expensive than the last: TF-IDF + logistic regression → BiLSTM + GloVe → frozen embeddings → fine-tuned transformer encoders (Ettin-400M as the main model, DeBERTa-v3-large as the strong baseline).
- Train each with class-weighted cross-entropy and compare them all on the same validation split using macro-F1.
- Predict the test set with the winner.

Live status: [PROGRESS.md](../../../PROGRESS.md). Fixed decisions: `CLAUDE.md`.

## Core resources (free)

If you only have a few hours, read these:
1. Karpathy, *A Recipe for Training Neural Networks*: http://karpathy.github.io/2019/04/25/recipe/
2. Hugging Face LLM Course, ch. 3 (fine-tuning with `Trainer`): https://huggingface.co/learn/llm-course/en/chapter3/3
3. Google ML Crash Course, accuracy, precision and recall: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
4. CS231n, *Neural Networks Part 3: Learning and Evaluation*: https://cs231n.github.io/neural-networks-3/

Each module lists its own sources. Resources were found via Perplexity on 2026-10-05; read them as supporting material.
