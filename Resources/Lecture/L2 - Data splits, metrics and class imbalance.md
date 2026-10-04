---
tags: [resources, lecture]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: sources below; project numbers from CLAUDE.md and PROGRESS.md
---
# L2 — Data splits, metrics and class imbalance

Part of [[MOC - ML Training Lecture]]. Previous: [[L1 - From backprop to a real training run]].

## 1. Why three datasets

| Split | Used for | Rule |
|---|---|---|
| **Training** | Fitting weights, through gradient descent | The model sees these labels |
| **Validation** | *Choosing* things: hyperparameters, epoch, model | Never trained on, but used to make decisions |
| **Test** | One final measurement | Touch once, at the end |

**Why validation isn't enough on its own.** Every decision you make by looking at validation scores slightly *overfits the validation set*. If you try 50 configurations and keep the best, its validation score is optimistically biased. The test set is the unbiased referee.

**In this project.** `test.json` has **no labels**, so the instructor holds the referee role and we can't compute test accuracy. Our validation split (20% of training) is the only honest performance estimate. That's why the protocol is frozen in `CLAUDE.md`: identical split and seed for every model, so comparisons are fair.

**Cross-validation (CV).** K-fold CV splits the data into K parts and trains K times, each time validating on a different part. You get K scores, and their average and spread are more reliable than one split. It costs K× the compute. That's cheap for TF-IDF and expensive for a 400M transformer. The brief lists CV as one reproducibility option.

## 2. Data leakage, the silent killer

**Leakage** is when information from the evaluation data reaches training, so your validation score lies. Typical forms:
- **Duplicates across splits.** If the same review is in train and validation, the model gets marked on a memorised answer. This is why we **drop the 216 duplicate reviews before splitting**.
- **Fitting preprocessing on all data.** A TF-IDF vocabulary or IDF weights computed on train + validation sneaks validation statistics into training. Fit transforms on **train only**, then apply them to validation and test. scikit-learn `Pipeline` enforces this.
- **Tuning on test.** Repeatedly checking the test score and adjusting turns it into a second validation set.

The 87 training reviews that also appear in `test.json` aren't leakage in our validation numbers. Training on the training set is allowed. They do mean those 87 test predictions are close to "seen" examples (useful for L8's error analysis).

## 3. Stratified splitting

With 85/15 classes, a random 20% split could by chance get 12% or 18% negatives. A **stratified** split keeps the class ratio identical in both parts (`train_test_split(..., stratify=y)`). Ours gives a training split of 839 negative / 4,909 positive, and a validation split of 1,437 reviews with about 210 negatives.

## 4. Metrics: why accuracy lies here

**Confusion matrix.** Treat "negative" as the class of interest (it's the rare one):

| | Predicted neg | Predicted pos |
|---|---|---|
| **Actually neg** | TP (true positive) | FN (false negative) |
| **Actually pos** | FP (false positive) | TN (true negative) |

- **Precision** = TP/(TP+FP): of the reviews we *called* negative, how many were?
- **Recall** = TP/(TP+FN): of the *truly* negative reviews, how many did we catch?
- **F1** = $2PR/(P+R)$, the harmonic mean, which is low if either P or R is low.
- **Macro-F1** = the average of the F1 for each class, computed with each class in turn as "positive". Both classes count equally, whatever their size.
- (Weighted F1 weights each class's F1 by its frequency. Micro-F1 equals accuracy for single-label problems. Neither protects the minority class.)

**The trap.** A "model" that always says *positive* scores **85.4% accuracy**. Its negative-class F1 is 0, positive F1 is 0.921, so macro-F1 is **0.46**. Accuracy calls that useless model "good"; macro-F1 exposes it. That's why the project selects models on **macro-F1** and reports accuracy, macro-F1 and negative-F1 together.

**How precise is a validation score?**
- Accuracy is a proportion, so its standard error is $\sqrt{p(1-p)/n}$. At $p=0.95$, $n=1437$ that's ≈ 0.0057, giving a 95% interval of about **±1.1 points**.
- So 95.8% vs 96.3% is *not* a meaningful difference.
- Negative-class metrics are noisier still: they rest on about 210 negatives, so each misclassified negative moves negative recall by about 0.5 points.

Report differences under about 1 point as ties. This is why the report draft says single-run differences under 1 point aren't claimed.

## 5. Class imbalance and class weights

**Problem.** With 85% positives, the cheapest way to reduce average loss is to lean towards "positive". Gradients from negatives are outnumbered about 6:1, so the model under-learns the minority class.

**Fix used here: class-weighted cross-entropy.** Multiply each example's loss by a weight for its class, $E = -w_y \log p_y$, using "balanced" weights:
$$w_c = \frac{N}{K\cdot N_c}$$
where $N$ = training examples, $K$ = number of classes and $N_c$ = examples of class $c$.

On our training split ($N=5748$, $K=2$): $w_{neg} = 5748/(2\cdot839) = 3.43$ and $w_{pos} = 5748/(2\cdot4909) = 0.585$.

Check: total weight per class is $839 \times 3.43 \approx 4909 \times 0.585 \approx 2874$. Both classes now pull equally on the gradient.

**Effect.** The decision boundary shifts towards predicting "negative" more often. Negative recall goes up, sometimes at a small cost in negative precision. Measured on our TF-IDF baseline: macro-F1 went from 0.766 to 0.841, negative-F1 from 0.58 to 0.73, and accuracy even rose slightly (0.911 to 0.920).

**Alternatives you should know (for the report's discussion).**
- **Oversampling** the minority class: repeat or duplicate negatives. Risks overfitting to the repeats.
- **Undersampling** the majority class: throw positives away. Wastes data.
- **Synthetic minority examples**: SMOTE for numeric features. For text, *generate* new negatives, which is the project's LLM-synthetic-negatives ablation and needs instructor approval.
- **Focal loss**: down-weights easy examples. Not used here.

Class weights were chosen because they're free (no data changes), principled (computed, not tuned) and supported everywhere (`class_weight="balanced"` in scikit-learn, `CrossEntropyLoss(weight=...)` in PyTorch).

## 6. Inference and the submission file (question d)

After choosing the final model on validation:
1. Run the model in eval mode on all 1,851 test reviews.
2. Take the argmax of the logits, which gives 0/1.
3. Write `submission.csv` in the **original test order**.

The column format is unconfirmed (an open question for the instructor). A sensible default is one row per review with the predicted `sentiments` value, plus an id or index column. Same preprocessing, same tokenizer, and no fitting on test.

> [!question]- Check yourself 1: you computed TF-IDF on the whole of `train.json` and then split. What's wrong?
> The IDF weights and vocabulary used statistics from the validation reviews. That's leakage. Split first, then fit the vectorizer on the training split only.

> [!question]- Check yourself 2: model A scores 96.1% accuracy and 0.90 macro-F1; model B scores 95.7% and 0.93 macro-F1. Which do you pick, and why?
> B. The 0.4-point accuracy gap is within noise (±1.1 points), while macro-F1 shows B handles the rare negative class much better. That's the selection metric we fixed in advance.

> [!question]- Check yourself 3: why compute class weights on the training split rather than the full dataset?
> Anything the model or its training uses should come only from training data. In practice the numbers barely differ (3.42 vs 3.43), but the discipline matters, and the report can state it.

## Sources and further study

- Google ML Crash Course, accuracy, precision and recall: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
- Google MLCC, imbalanced datasets: https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets
- scikit-learn, model evaluation (F1 averaging options): https://scikit-learn.org/stable/modules/model_evaluation.html
- scikit-learn, common pitfalls (leakage): https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn, cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- imbalanced-learn user guide (resampling): https://imbalanced-learn.org/stable/user_guide.html
- King & Zeng 2001, *Logistic Regression in Rare Events Data* (the basis of balanced weights): https://gking.harvard.edu/files/0s.pdf
