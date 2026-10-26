---
tags: [resources, lecture]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: sources below; data observations from CLAUDE.md
---
# L8 — Error analysis, domain shift and noisy labels (questions e, g, h)

Part of [[MOC - ML Training Lecture]]. Previous: [[L7 - How to choose a model and method]].

## Part A: error analysis (question e)

**What it is.** Look at individual mistakes, group them into causes and count each cause. A single score tells you *how much* the model errs; error analysis tells you *why*. That's what "strengths and weaknesses" means in (e).

**Method (Ng, *Machine Learning Yearning*):**
1. Take the validation predictions and split them into four buckets: correct negatives, correct positives, false negatives (true negative predicted positive) and false positives.
2. Read about 50 errors and tag each with a cause. Likely causes for our data:
   - **Mixed sentiment**: "looks great but fell apart".
   - **Sarcasm or irony.**
   - **Negation or contrast across sentences.**
   - **Label noise**: the text reads clearly positive but is labelled 0 (labels appear to come from star ratings).
   - **Very short or uninformative text.**
   - **Truncation**: the decisive part comes after 512 tokens (rare: 0.33%).
3. Count the tags to get a table: "38% mixed, 25% likely mislabelled …". This tells you what would actually help.
4. Use the **model's confidence** (softmax probability):
   - Confident and correct cases show strengths.
   - **Confident but wrong** cases are often label errors.
   - Low-confidence cases are genuinely ambiguous.

**The test-set catch.** (e) says "samples in the test set", but test labels are hidden, so you can't know which test predictions are wrong. Honest options:
- Use the **87 test reviews that also appear in `train.json`**, whose labels are known.
- Pick test reviews whose sentiment is unambiguous to a human reader, and judge those.
- Support either with validation-set error statistics.

Say explicitly in the report which you did and why.

**A good (e) case study** quotes the review, gives the true and predicted label with confidence, then explains the mechanism. For example: "attention to 'but' let the model weigh the second clause" (strength), or "the label says positive but the text complains, so this is likely a rating/text mismatch, not a model error" (a weakness of the data, not the model).

## Part B: domain shift, hotel reviews with no ratings (question g)

**What changes.** Training data comes from a source distribution (product reviews); deployment data from a target distribution (hotel reviews).
- **Vocabulary and aspects change**: "battery", "shipping" and "broke" become "staff", "location", "noisy room" and "check-in".
- **Label meaning may shift** too: what counts as negative for a hotel.

This is **domain shift**, and accuracy usually drops: modestly for general sentiment cues ("terrible", "loved it"), more for domain-specific ones ("thin walls"). Without labels **you can't even measure the drop**, so evaluation is part of the problem.

**How our system would perform.**
- The fine-tuned transformer transfers better than TF-IDF, because pretraining covers hotel language and sentiment words generalise.
- TF-IDF's vocabulary simply lacks hotel terms.
- Class balance may also differ (the 85/15 prior and class weights may be wrong for hotels).

**How to adapt it, with no labels:**
1. **Zero-shot models.** Sentiment-tuned models (SiEBERT), NLI-based zero-shot, GLiClass, LLM prompting. They need no target labels. Our zero-shot runs give evidence of how well they work on product reviews.
2. **Domain-adaptive pretraining (DAPT).** Continue masked-LM pretraining of our encoder on lots of **unlabelled** hotel text, then reuse our product-trained classifier. Gururangan et al. show consistent gains from continuing pretraining on in-domain text.
3. **Few-shot labelling plus SetFit.** Hand-label a small set (around 8–64 per class) and train with SetFit (contrastive fine-tuning of a sentence encoder plus a light head). This works with tiny label counts.
4. **Self-training / pseudo-labelling.** Label hotel reviews with the current model, keep only high-confidence predictions, retrain, and repeat. Risk: confirmation bias.
5. **Always**: hand-label a small hotel **test set** to measure actual performance.

## Part C: noisy labels (question h)

**What happens.**
- Large networks can **memorise anything**, even random labels (Zhang et al. 2017).
- With noisy labels, training eventually fits the wrong labels too, and generalisation degrades.
- Networks tend to **learn clean, consistent patterns first and memorise noise later** (Arpit et al. 2017). So noisy-label training typically shows validation improving, then *declining* as memorisation begins. Early stopping partly helps.
- Our own data already has some noise (star-rating-derived labels, about 2–3% suspected errors), so (h) can reference real examples.

**Three approaches (the brief asks for three; the project's choice):**
1. **Confident learning / cleanlab** (*clean the data*).
   - Get out-of-sample predicted probabilities via cross-validation.
   - Estimate per-class confidence thresholds, then build the "confident joint" (labelled *i* but confidently *j*) to estimate *which* examples are mislabelled.
   - Remove, relabel or down-weight them, then retrain.
   - Northcutt et al. also found that famous benchmark test sets contain label errors (labelerrors.com).
2. **Co-teaching** (*robust training procedure*). Train two networks simultaneously. Each picks its **small-loss** examples in each mini-batch (likely clean, since clean patterns are learned first) and trains its *peer* on them. The two networks filter different errors, so mistakes don't self-reinforce (Han et al. 2018).
3. **Label smoothing** (*robust targets*). Replace hard targets 1/0 with 1−ε / ε (for example 0.9 / 0.1). The model is never pushed to infinite confidence, so a wrong label does less damage and outputs are better calibrated (Müller et al. 2019).

Other options for discussion: robust losses (symmetric or generalised CE), noise-transition modelling, early stopping, or collecting multiple annotations and aggregating them.

**How to frame (h).** First explain *how* our algorithm degrades: memorisation, misleading validation if validation is also noisy, and a model confidently replicating annotator errors. Then give the three approaches, each with its mechanism, an example on our data and its cost.

> [!question]- Check yourself 1: why are "confident but wrong" validation predictions especially interesting?
> They're the best candidates for label errors rather than model errors. The model has learned a consistent pattern, and the label contradicts it. That's exactly the signal cleanlab formalises.

> [!question]- Check yourself 2: in (g) you can't compute accuracy on hotel reviews. What's the minimum you'd do before deploying?
> Hand-label a small, representative hotel test set (a few hundred reviews) and measure macro-F1 there. Without it, every adaptation method is unverified.

> [!question]- Check yourself 3: how does co-teaching use the "clean patterns are learned first" observation?
> Early in training, clean examples have small loss and noisy ones have large loss. Each network trains its peer only on its small-loss subset, so the noisy examples are mostly excluded from updates.

## Sources and further study

- Andrew Ng, *Machine Learning Yearning* (error analysis chapters): https://info.deeplearning.ai/machine-learning-yearning-book
- Northcutt et al. 2021, *Confident Learning*: https://arxiv.org/abs/1911.00068 · cleanlab docs: https://docs.cleanlab.ai/ · label errors browser: https://labelerrors.com/ (paper: https://arxiv.org/abs/2103.14749)
- Han et al. 2018, *Co-teaching*: https://arxiv.org/abs/1804.06872
- Müller, Kornblith & Hinton 2019, *When Does Label Smoothing Help?*: https://arxiv.org/abs/1906.02629
- Zhang et al. 2017, *Understanding Deep Learning Requires Rethinking Generalization*: https://arxiv.org/abs/1611.03530
- Arpit et al. 2017, *A Closer Look at Memorization in Deep Networks*: https://arxiv.org/abs/1706.05394
- Gururangan et al. 2020, *Don't Stop Pretraining* (DAPT): https://arxiv.org/abs/2004.10964
- HF blog, SetFit: https://huggingface.co/blog/setfit · zero-shot classification: https://huggingface.co/blog/zero-shot-classification · Davison's explainer: https://joeddav.github.io/blog/2020/05/29/ZSL.html
