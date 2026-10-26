# Report draft — Question (e) — Correct and incorrect test predictions

Submitted model: Ettin-400M encoder (`ettin400m_final`). The Ettin + Qwen3-8B cascade (`src/cascade.py`) is used here
as an analysis tool, not as the submission.

## (e) Strengths and weaknesses on test samples

**How we judge test predictions without test labels**
- `test.json` has no labels, so no test prediction can be marked right or wrong for certain.
- We therefore combine two sources:
  1. **Validation errors** (1,437 labelled reviews from the same data) to measure *which kinds* of reviews the model gets wrong.
  2. **Test reviews whose label is clear from the text**, examined with sentence occlusion (remove one sentence, re-score) to see *why* the model decided.
- Sentence occlusion numbers below come from single-review runs; they can differ from the batched submission run by a
  few hundredths on borderline reviews.

**What the test predictions look like**
- 1,851 test reviews: 258 predicted negative (13.9%; training set 14.6%).
- 96.3% are predicted with confidence ≥ 0.99; only 69 fall below.

**Error profile on validation (known labels)**

| | Count |
|---|---|
| Errors (of 1,437) | 44 (accuracy 0.969, macro-F1 0.940) |
| Negative reviews predicted positive / positive predicted negative | 18 / 26 |
| Errors containing a contrast word ("but", "however", "unfortunately", …) | 64%, vs 38% of all reviews |
| Errors made with confidence > 0.99 | 21 |
| Reviews wrong in all 9 Ettin training runs | 22 |

The errors are concentrated in **mixed reviews**: praise and complaint in the same review, where the label follows the
customer's star rating rather than the overall tone. Some are plain label noise (e.g. "These are NOT as they appear in
the pic … NOT cool", labelled positive).

### Case 1 — Strength: reading a mild complaint in context (test id 276)

> "These really do stay on. The cuff gets tighter after a run in the dryer - thats why I didn't give 5 stars. Dry on a low temp if at all"

| Model | P(positive) | Prediction |
|---|---|---|
| Ettin-400M (submitted) | 1.000 | positive |
| TF-IDF + logistic regression (1–2-grams, balanced weights) | 0.28 | negative |

Sentence occlusion (Ettin):

| Sentence | Scored alone | Review without it |
|---|---|---|
| "These really do stay on." | 0.986 | 0.901 |
| "The cuff gets tighter after a run in the dryer" | 0.889 | 0.047 |
| "thats why I didn't give 5 stars." | 0.283 | 0.996 |
| "Dry on a low temp if at all" | 0.521 | 1.000 |

- The reviewer says they gave fewer than 5 stars because of one minor issue, which almost certainly means 4 stars.
  Reviews like this are labelled **positive** in this dataset, so the prediction is very likely correct.
- Bag-of-words sees "didn't", "tighter" and "5 stars" as separate features and leans negative.
- Ettin reads the sentences together. "I didn't give 5 stars" alone scores negative (0.283), but next to a small,
  specific complaint the review scores 1.000. Removing the complaint sentence drops it to 0.047, because "didn't give 5
  stars" without a reason reads as a real criticism.
- **Strength shown:** contextual embeddings weigh a complaint by how serious it is, not by its words, and the model has
  learned this dataset's convention that a minor complaint still means a positive rating.

### Case 2 — Weakness: praise first, dealbreaker second (test id 82)

> "These are adorable pumps, it's too bad I couldn't even put my foot down to walk in them. The heel is much too long"

| Model | P(positive) | Prediction |
|---|---|---|
| Ettin-400M (submitted) | 0.974 | positive |
| Ettin + Qwen3-8B cascade | 0.000 (Qwen) | negative |

Sentence occlusion (Ettin):

| Sentence | Scored alone | Review without it |
|---|---|---|
| "These are adorable pumps, it's too bad I couldn't even put my foot down to walk in them." | 0.993 | 0.007 |
| "The heel is much too long" | 0.007 | 0.993 |

- The shoes could not be walked in, so the customer most likely gave a low rating: the submitted prediction is very
  likely **wrong**.
- The first sentence alone scores 0.993: "adorable" outweighs "couldn't even put my foot down to walk in them" in the
  same sentence. Only the short second sentence pulls the score down, and not far enough.
- The model is still 97% confident. On validation, 21 of 44 errors were made above 99% confidence, so the model's
  confidence does not reliably flag its mistakes.
- **Weakness shown:** in mixed reviews, strong praise words can outweigh a complaint that makes the product unusable,
  and the model stays overconfident when it does so.

**How the cascade handles this case**
- Ettin's confidence (0.974) is below the cascade's 0.99 cut-off (chosen on training data), so the review goes to
  Qwen3-8B. Qwen is not trained; its prompt holds the 8 most similar training reviews with their labels.
- Those 8 reviews are split 5 positive / 3 negative (e.g. "look great … not good for walking much because of the heel",
  labelled positive). A majority vote would still say positive; Qwen reads the review against them and answers
  negative.
- On validation the cascade fixes 13 of Ettin's errors and introduces 8: macro-F1 0.940 → 0.946, not statistically
  significant (McNemar p = 0.38). It sends only 59 of 1,437 validation reviews (4%) and 69 of 1,851 test reviews to
  Qwen.

**Limit: when the text does not decide the label (test id 1479)**

> "I do not have large ankles or calves, but these boots fit TIGHT!! I ordered a 1/2 size too large but I was still barely able to zip them up."

- Ettin is at chance (P(positive) ≈ 0.36–0.48 across runs). All 8 retrieved training reviews are positive "runs
  small, but love them" boot reviews, and Qwen follows them (1.000, positive).
- A reader cannot tell whether this customer gave 2 stars or 4. The true label depends on a rating the text does not
  contain, so no model can be reliably right here. This is the irreducible part of the error (an estimated 22 or so
  validation reviews that every model gets wrong).

**Summary**

| | Strength | Weakness |
|---|---|---|
| Ettin-400M | Reads negation and contrast in context; learned the dataset's star-rating convention (case 1) | Strong praise can outweigh a dealbreaker; overconfident on its errors (case 2) |
| Cascade (analysis) | Fixes some mixed reviews by comparing them with similar labelled training reviews (case 2) | Small, non-significant gain; follows one-sided neighbours when the text is ambiguous (id 1479) |

Evidence: `outputs/runs/ettin400m_final/test_check_with_probs.csv` (test probabilities, same model as `submission.csv`),
`outputs/runs/cascade_qwen3-8b/` (cascade metrics, LLM cache, test predictions), `docs/ettin_gaps_and_hybrids.md`
(all cascade and specialist experiments).
