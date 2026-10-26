---
tags: [resources, lecture]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: sources below
---
# L5 — Training in practice: hyperparameters and debugging (questions b and c)

Part of [[MOC - ML Training Lecture]]. Previous: [[L4 - Classifier architectures]].

## 1. Parameters vs hyperparameters

- **Parameters** are the weights, learned by gradient descent.
- **Hyperparameters** are everything *you* set before training: learning rate, batch size, epochs, schedule, weight decay, dropout, max length, model size. They're chosen by **validation performance**, never by training loss and never by test.

The brief's step 4 says to try "feature dimension, weights, initialization, and learning rate". Question (c) asks how and why you chose them.

## 2. The hyperparameters that matter, in order of importance

**1. Learning rate η (by far the most important).**
- Too high: the loss spikes or goes to NaN, or a pretrained model's knowledge is wrecked (catastrophic forgetting).
- Too low: slow progress, and the model may underfit within your epoch budget.
- Typical values: scratch-trained small nets with Adam about 1e-3; **fine-tuning pretrained transformers 1e-5 to 5e-5** (the pretrained weights are already good; you nudge them).

**2. Learning-rate schedule.**
- **Warm-up**: grow η from 0 to its peak over the first ~5–10% of steps. Adam's moment estimates are unreliable at the start, and the random head produces large gradients. Warm-up stops those early steps from damaging pretrained weights.
- **Decay**: then shrink η, linearly or with a cosine curve, towards 0. Big steps early to make progress, small steps late to settle into a minimum.
- "Linear warm-up then linear decay" is the standard BERT recipe and the HF `Trainer` default shape.

**3. Batch size.**
- Larger batches give smoother gradient estimates and better GPU throughput, but cost more memory and give fewer updates per epoch.
- Rule of thumb: if you change batch size, re-tune η.
- **Gradient accumulation** fakes a big batch: run 2 micro-batches of 8, add their gradients, and update once. That's an effective batch of 16, which L1 called "mini-batch updating", done in pieces to save memory.

**4. Number of epochs.**
- Fine-tuning a pretrained model on a few thousand examples usually peaks within **2–4 epochs**. More epochs mean memorising training data, so validation gets worse.
- Scratch-trained models (BiLSTM) need more, around 10–30.
- Don't guess: **evaluate each epoch and keep the best checkpoint** (§4).

**5. Regularisation: methods that reduce overfitting.**
- **Weight decay** (AdamW): pulls weights towards 0. Typically 0.01 for transformers.
- **Dropout**: randomly zero ~10% of activations in training, so the model can't rely on any single unit. It acts like averaging many sub-networks, and it's turned off at evaluation (`model.eval()`).
- **Early stopping or checkpoint selection**: stop at the best validation score.
- **More or augmented data.** The synthetic-negatives ablation is one form.

**6. Initialisation.**
- Scratch nets: small random weights (the lecture's "small random values"). Libraries use Xavier/Kaiming scaling automatically.
- Pretrained models: the *best possible* initialisation, which is the whole point of transfer learning. Only the head is random.
- The random seed controls this, which is why seeds matter for reproducibility.

## 3. Underfitting vs overfitting: reading the curves

Plot **training loss and validation loss per epoch**:

| Pattern | Diagnosis | Action |
|---|---|---|
| Both high, both flat | **Underfitting**: model too weak, η too low, or a bug | Bigger model or features, higher η, train longer, check the pipeline |
| Training ↓, validation ↓ | Still learning | Keep going |
| Training ↓, validation ↑ (gap widens) | **Overfitting**: memorising | Stop earlier, regularise, get more data |
| Loss spikes or NaN | η too high, or numerical issues | Lower η, add warm-up, check bf16/fp16 |
| Loss stuck at ~0.693 (2 classes) | Learning nothing | Labels shuffled or misaligned, η wrong, frozen weights |

The lecture's line "small training error alone does not establish generalization" *is* this table.

## 4. Checkpoint selection (early stopping)

Evaluate on validation after every epoch, compute **macro-F1** (the selection metric fixed in advance, L2) and save the model whenever it improves. At the end, load the best. In HF this is `eval_strategy="epoch"`, `save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="macro_f1"`.

Choosing the epoch on validation is itself a decision, so the reported validation number is slightly optimistic. That's acceptable; say so honestly.

## 5. Hyperparameter search

- **Budget reality**: a 400M fine-tune takes tens of minutes or more, so you can't try hundreds of settings. The literature shows **η matters most**, so sweep η over a few values (for example {1e-5, 2e-5, 3e-5}) and take the rest from established defaults. That's exactly the plan in `docs/report_draft_abc.md`.
- **Grid search**: try every combination. Fine for 1–2 hyperparameters. **Random search** beats grids when many hyperparameters matter, because it covers more distinct values of the important ones.
- **Change one thing at a time** against a fixed baseline, so you know what caused a change. This is ablation discipline (L7).
- Cheap models (TF-IDF) can afford proper sweeps (C, n-gram range, min_df) with cross-validation.

## 6. Karpathy's recipe: how practitioners actually work

1. **Become one with the data.** Read 50–100 reviews. Look at lengths, duplicates and odd labels. (Done: the data facts in `CLAUDE.md` came from this.)
2. **Set up end-to-end training and evaluation with a dumb baseline first.** Majority class (0.854 accuracy), then TF-IDF. Now you know what "good" means.
3. **Sanity checks.**
   - Initial loss ≈ ln 2 = 0.693.
   - **Overfit one tiny batch**: train on 16 examples until training loss ≈ 0. If the model can't memorise 16 examples, the pipeline is broken.
   - Look at a decoded input batch: are the labels aligned with the right texts?
4. **Overfit, then regularise.** First get a model big enough to fit the training data well, then add regularisation to close the train–validation gap.
5. **Tune**, starting with η.
6. **Squeeze out the rest**: ensembles, longer training. This project consciously stops before this step (the owner dropped soups and multiple seeds).

## 7. How to write question (c)

Use a table where every row says *value → how it was determined → why*. There are four legitimate "how"s:
- **Swept on validation**: learning rate.
- **Taken from the literature, with a citation**: warm-up 10%, weight decay 0.01, 2–4 epochs.
- **Derived from the data**: max length 512 from token statistics; class weights from class counts.
- **Forced by hardware**: batch 8 × 2 accumulation, 8-bit AdamW, bf16, model size.

Add the sweep results table once it exists. The marker wants to see that nothing was arbitrary.

> [!question]- Check yourself 1: training macro-F1 is 0.99 and validation is 0.88, and validation dropped after epoch 2. What happened and what do you do?
> Overfitting after epoch 2. Use the epoch-2 checkpoint (checkpoint selection) and optionally strengthen regularisation or lower η. Don't keep training.

> [!question]- Check yourself 2: why sweep only the learning rate, not batch size, warm-up and weight decay too?
> Compute budget. Each run is expensive, and the fine-tuning literature (Mosbach et al. 2021; Sun et al. 2019) shows η dominates while the others have robust defaults. A full grid would cost dozens of runs for little expected gain.

> [!question]- Check yourself 3: what does "overfit a single batch" test, and what result means "bug"?
> It tests that the model, the loss and the optimizer are wired correctly. A healthy setup drives training loss on 16 examples close to 0 within a few hundred steps. If the loss stays near 0.693, something is broken: labels, frozen parameters, a wrong loss input, or an η that's far off.

## Sources and further study

- Karpathy, *A Recipe for Training Neural Networks*: http://karpathy.github.io/2019/04/25/recipe/
- CS231n, *Neural Networks Part 3: Learning and Evaluation* (sanity checks, loss curves, hyperparameter search): https://cs231n.github.io/neural-networks-3/
- Google, *Deep Learning Tuning Playbook*: https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook · learning-rate section: https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/learning-rate
- Google MLCC, interpreting loss curves: https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves
- D2L, weight decay: https://d2l.ai/chapter_multilayer-perceptrons/weight-decay.html · dropout: https://d2l.ai/chapter_multilayer-perceptrons/dropout.html
- Mosbach et al. 2021, *On the Stability of Fine-tuning BERT*: https://arxiv.org/abs/2006.04884
- Sun et al. 2019, *How to Fine-Tune BERT for Text Classification?*: https://arxiv.org/abs/1905.05583
