# Ettin gaps, hybrid options, and what industry does (research, 2026-10-26)

Research question: where is Ettin-400M weak for this task, which hybrid methods are backed by evidence, and how do
benchmark/industry systems reach very high accuracy? Includes one measurement on our own validation predictions.

## 1. Measured on our runs (no new training)

Nine Ettin-400M runs on the same 1,437-review validation split (baseline, final, normtext, lr 1e-5, lr 3e-5, 4 epochs,
neg weight 1.5 / 2.0, balanced weights). Script: averaged `p_pos` from each run's `val_predictions.csv`.

| Check | Result |
|---|---|
| Errors per run | 39–54 |
| Reviews wrong in **all 9** runs | 22 |
| Average of all 9 runs' probabilities | acc 0.9715, macro-F1 **0.943** (best single run 0.947) |
| `ettin400m_final` errors made with confidence > 0.99 | 21 of 44 |
| `ettin400m_final` reviews below 0.99 confidence | 59 (holding 23 of the 44 errors) |

Meaning:
- Ettin runs make the **same mistakes**, so averaging Ettins (different configs, same model) adds nothing. Any ensemble
  gain has to come from a *different* model family.
- Half the errors are made with near-certainty. A confidence-based fallback (cascade to an LLM) cannot even see them.
  Reading them, most are the star-rating label convention or plain mislabels (e.g. "NOT as they appear in the pic …
  NOT cool" labelled positive). Rough ceiling: ~22 errors are hard for any model → ~98.5% accuracy at best.

### Follow-up: does a different model family make different mistakes?

SiEBERT zero-shot (bf16, max length 512) on the same 1,437 reviews vs `ettin400m_final`:

| Pair | Errors | Shared errors | Correlation of error indicators |
|---|---|---|---|
| Ettin final vs Ettin baseline (same model, two runs) | 44 / 39 | 29 | 0.69 |
| Ettin final vs SiEBERT zero-shot | 44 / 59 | 31 | 0.59 |

Average of Ettin + SiEBERT zero-shot: macro-F1 0.935 (equal weights) / 0.937 (0.7 Ettin), both **below** Ettin alone
(0.940). SiEBERT is only a little more diverse than another Ettin run, and it is weaker, so it adds more new errors (28)
than it can fix (13). The shared errors are the dataset's hard core, not a model weakness. This weakens the case for
an ensemble: build one only if a fine-tuned second model shows both similar accuracy **and** clearly lower error
correlation (well under ~0.6).

### Follow-up: a specialist for the ambiguous reviews (router + specialist)

Router = rank validation reviews by Ettin's uncertainty (1 − max prob). Error-detection AUROC 0.86; a TF-IDF error
detector trained on the 5-fold OOF errors adds nothing (0.86; text alone 0.73).

| Routed (most uncertain) | Ettin errors inside | Negatives share | Acc on routed: Ettin final / other Ettin run / SiEBERT zero-shot | Ceiling if specialist were perfect |
|---|---|---|---|---|
| top 30 | 10 / 44 | 0.47 | 0.67 / 0.67 / 0.63 | macro-F1 0.953 |
| top 60 | 23 / 44 | 0.47 | 0.62 / 0.72 / 0.60 | 0.971 |
| top 100 | 30 / 44 | 0.48 | 0.70 / 0.74 / 0.65 | 0.981 |

Finding the hard reviews works; answering them is the bottleneck. Every model we have scores 60–74% on them, where
chance is ~50%. The hand review of the 146 OOF-hard training reviews found 64 (44%) with no clear sentiment and 32
(22%) whose label contradicts the text, leaving ~50 learnable examples. A specialist trained only on hard cases would
learn mostly label noise (boosting / hard-example mining overfits random label noise: Long & Servedio 2010). Bar to
clear: > 70% on the top-100; 80% would fix ~10 errors (+0.7 acc). Better design: route to a strong generalist trained
on all data (e.g. DeBERTa) and fix the routing size on OOF data, not validation.

### Follow-up: off-the-shelf "ambiguity specialists"

No model or benchmark targets mixed/ambiguous reviews with star-derived labels (searched 2026-10-26). Closest match:
star-rating predictors trained on Amazon reviews, since our labels look star-derived. Each model's class probabilities
were mapped to 0/1 by a LogisticRegression fitted on the training split only, then used on Ettin's k most uncertain
validation reviews (Ettin final alone: macro-F1 0.9396; accuracy on top 30/60/100 = 0.67 / 0.62 / 0.70).

| Specialist (training data) | Full val macro-F1 alone | Acc top 30 / 60 / 100 | Cascade macro-F1 top 30 / 60 / 100 |
|---|---|---|---|
| nlptown/bert-base-multilingual-uncased-sentiment (1–5★ product reviews) | 0.917 | 0.57 / 0.63 / 0.73 | 0.935 / 0.940 / 0.942 |
| Nav772/roberta-amazon-reviews-5star (1–5★ Amazon, 20k) | 0.925 | 0.67 / 0.65 / 0.72 | 0.939 / 0.942 / 0.942 |
| RashidNLP/Amazon-Deberta-Base-Sentiment (neg/neu/pos Amazon, 1M) | 0.927 | 0.67 / 0.60 / 0.72 | 0.939 / 0.937 / 0.942 |
| LiYuan/amazon-review-sentiment-analysis (1–5★ Amazon) | 0.915 | 0.53 / 0.62 / 0.71 | 0.933 / 0.938 / 0.939 |
| siebert/sentiment-roberta-large-english (15 sentiment sets) | 0.929 | 0.63 / 0.63 / 0.68 | 0.938 / 0.941 / 0.937 |

Best cascade +0.003 macro-F1 (about 2 reviews), within noise, and only at a k picked after seeing validation. No
specialist beats Ettin on the hard reviews by a usable margin. All five also use outside labelled data (instructor
question). Literature agrees: 3-star/mixed reviews are the hardest case for humans and LLMs alike
([arXiv 2506.13313](https://arxiv.org/html/2506.13313v1)); LLMs beat fine-tuned encoders on some sarcasm sets but not
consistently on implicit sentiment ([arXiv 2408.11319](https://arxiv.org/html/2408.11319v2)).

### Follow-up: local LLM with nearest-neighbour examples (KATE-style)

Qwen3-8B, 4-bit NF4 (fits the 8 GB laptop GPU), thinking disabled; score = next-token probability of "positive" vs
"negative". System prompt states labels come from the customer's star rating. Examples come from the training split
only, retrieved with Qwen3-Embedding-0.6B (cosine, 8 per prompt, most similar last). Same 100 most uncertain
validation reviews as the specialist table.

| Condition | Acc top 30 / 60 / 100 | Cascade macro-F1 top 30 / 60 / 100 | On top 100: fixes / breaks Ettin | McNemar p (top 100) |
|---|---|---|---|---|
| Ettin final alone | 0.67 / 0.62 / 0.70 | 0.940 | — | — |
| Qwen3-8B zero-shot | 0.67 / 0.67 / 0.70 | 0.940 / 0.945 / 0.942 | 18 / 18 | 1.00 |
| Qwen3-8B, 8 random examples | 0.70 / 0.68 / 0.74 | 0.941 / 0.945 / 0.945 | 15 / 11 | 0.56 |
| **Qwen3-8B, 8 nearest examples** | **0.80 / 0.72 / 0.79** | **0.945 / 0.947 / 0.951** | **17 / 8** | 0.11 |

- First specialist that beats Ettin on the hard reviews at every routing size (+9 net reviews at top 100, +1.2
  macro-F1), but not yet significant (p = 0.11) and the top-100 size was not fixed in advance.
- Zero-shot judges text sentiment: it calls 74% of the routed reviews negative (true negatives 0.96, true positives
  0.46). Similar labelled examples teach it the star-rating convention (true positives 0.46 → 0.83); random examples
  only partly do. So the gain comes from retrieval, not from the LLM itself.
- Speed: only the routed reviews (7% at top 100 of 1,437) reach the LLM; Ettin handles the rest.
- Next check before using it for the submission: repeat on the 5-fold out-of-fold training predictions (more hard
  reviews, independent of validation) and fix the routing size there.
- Script: scratchpad `llm_knn.py` (not yet in `src/`).

### Follow-up: cascade with the cut-off fixed on training data (`src/cascade.py`, `src/cascade_embed.py`)

Route a review when Ettin's confidence < threshold; candidate thresholds 0.9 / 0.99 / 0.995 / 0.999; the threshold is
chosen by macro-F1 on the 5-fold out-of-fold training predictions (Ettin OOF alone 0.949), never on validation.
Runs: `outputs/runs/cascade_qwen3-8b/`, `outputs/runs/cascade_embed/` (metrics.json, test submissions).

| Specialist | Threshold chosen on OOF | Val routed | Val macro-F1 (Ettin 0.940) | Fixed / broke | McNemar p | Test routed / changed |
|---|---|---|---|---|---|---|
| **Qwen3-8B + 8 nearest examples** | 0.99 (OOF 0.953) | 59 | **0.946** | 13 / 8 | 0.38 | 69 / 26 |
| Similarity vote, 8 nearest (no LLM) | 0.9 (OOF 0.946) | 27 | 0.937 | 8 / 9 | 1.00 | 27 / 13 |
| LogReg on Qwen3-Embedding vectors | 0.9 (OOF 0.948) | 27 | 0.930 | 6 / 12 | 0.24 | 27 / 7 |

- The Qwen cascade holds up under the honest protocol: +0.6 macro-F1 on validation, consistent with +0.4 on OOF, but
  not significant. The earlier 0.951 used a routing size picked after seeing validation.
- The neighbours' labels alone do not carry the gain: a vote over the same 8 reviews scores 0.71 on the 100 most
  uncertain (Qwen 0.79) and the cascade loses. Qwen's reading of the review plus the examples is what helps.
- Cost: Qwen sees 59 of 1,437 validation reviews (4%) and 69 of 1,851 test reviews.

### Follow-up: the Ettin decoder twin

**Fine-tuned** (`ettin_dec400m_lr2e-5`, same protocol as the encoder: lr 2e-5, 3 epochs, no weights, last-token
pooling, 13.2 min, 5.7 GB): acc 0.962, macro-F1 **0.924**, neg-F1 0.870 vs encoder `ettin400m_final` 0.940
(McNemar: only encoder wrong 9, only decoder wrong 20, p = 0.061). Error correlation with the encoder 0.70 (35 of 44
encoder errors shared), same as two encoder runs. Average of the two 0.932. As a specialist on the 30 / 60 / 100 most
uncertain reviews: acc 0.63 / 0.58 / 0.66 vs encoder 0.67 / 0.62 / 0.70 → cascade 0.937 / 0.936 / 0.933. Matches the
Ettin paper: same data and size, the encoder wins classification; the twin makes the same mistakes.

**In-context, not fine-tuned** (base LMs, same prompt idea, routed reviews and retrieved examples as the Qwen test):

| Model | Acc top 30 / 60 / 100 | Cascade macro-F1 top 30 / 60 / 100 | Fixes / breaks (top 100) | Predicted negative |
|---|---|---|---|---|
| Ettin encoder alone | 0.67 / 0.62 / 0.70 | 0.940 | — | — |
| dec-400m zero-shot | 0.60 / 0.60 / 0.69 | 0.936 / 0.936 / 0.936 | 14 / 15 | 0.37 |
| dec-400m 8 random | 0.60 / 0.60 / 0.64 | 0.937 / 0.939 / 0.934 | 13 / 19 | 0.70 |
| dec-400m 8 nearest | 0.60 / 0.65 / 0.69 | 0.937 / 0.943 / 0.940 | 16 / 17 | 0.67 |
| dec-1b zero-shot | 0.57 / 0.58 / 0.60 | 0.934 / 0.933 / 0.920 | 17 / 27 | 0.12 |
| dec-1b 8 random | 0.50 / 0.55 / 0.63 | 0.934 / 0.936 / 0.934 | 15 / 22 | 0.81 |
| dec-1b 8 nearest | 0.47 / 0.53 / 0.61 | 0.933 / 0.935 / 0.931 | 14 / 23 | 0.83 |
| Qwen3-8B 8 nearest (above) | 0.80 / 0.72 / 0.79 | 0.945 / 0.947 / 0.951 | 17 / 8 | 0.45 |

Ettin decoders are small base LMs (no instruction tuning): with examples in the prompt they swing to heavily
negative answers instead of learning the convention. In-context learning from retrieved examples needs a larger,
instruction-tuned model.

## 2. Gaps / flaws of Ettin-400M for this task

1. **MLM-only pretraining.** DeBERTa-v3 adds replaced-token detection (ELECTRA-style), which is better for
   classification. The ModernBERT authors list this as their own limitation and suggest RTD would help. GLUE average:
   DeBERTa-v3-large 91.4, Ettin-400M 90.8, ModernBERT-large 90.4.
   [ModernBERT](https://arxiv.org/abs/2412.13663), [Ettin](https://arxiv.org/abs/2507.11412)
2. **Sample efficiency and stability.** A controlled study (same training data for both) found DeBERTa-v3 more
   sample-efficient and better overall than ModernBERT; ModernBERT showed learning-rate sensitivity and some runs that
   diverged (NaN loss). Ettin uses the ModernBERT architecture. Our 5.7k training rows are the small-data regime where
   this matters; our own runs spread 0.925–0.947 across 11 configurations, and lr 3e-5 scored 0.901 after epoch 1.
   [Antoun et al., IJCNLP 2025](https://aclanthology.org/2025.ijcnlp-long.164.pdf) (French models; transfer to English
   is likely but not shown)
3. **Its advantages do not apply here.** Ettin/ModernBERT's selling points are speed (unpadding + FlashAttention 2) and
   8k context. Our reviews are median 54 tokens and Windows runs use SDPA, so we pay no accuracy price but get no
   accuracy benefit either.
4. **Not sentiment-specialised.** General web pretraining only. SiEBERT (RoBERTa-large already trained on 15 sentiment
   datasets incl. product reviews) scores 0.921 macro-F1 with **zero** training on our data.
5. **Overconfident errors.** 97.6% of predictions sit at P > 0.95 or < 0.05, and 21 of 44 errors are > 0.99. Good for
   accuracy, bad for any method that relies on uncertainty (cascades, abstention, active relabelling).
6. **Single-run noise > most technique gains.** Validation has 210 negatives; one config's spread is ±1 macro-F1 point,
   larger than the typical published gain of training tricks (LLRD, AWP, multi-sample dropout: +0.3–0.9).

## 3. Hybrid methods: evidence and expected value here

| Method | Evidence | Expected value for us | Verdict |
|---|---|---|---|
| **Heterogeneous ensemble**: Ettin-400M + DeBERTa-v3-large + fine-tuned SiEBERT, average probabilities | Ensembles of different transformer backbones beat the best single model on 6 of 8 sentiment datasets (SST-2, IMDb, Yelp …) ([Springer 2025](https://link.springer.com/article/10.1007/s41060-025-00963-0)); different pretraining → less correlated errors | Uncertain. Ettin + SiEBERT zero-shot averaging *lost* 0.4–0.5 macro-F1 (§1 follow-up). Worth testing only after the single DeBERTa run, if its errors correlate clearly less with Ettin's | **Test, don't assume** |
| **TAPT**: continue masked-LM on our training reviews (text only) before fine-tuning | Gururangan et al. 2020: task-adaptive pretraining helps review tasks, ~+0.5 on IMDb, more on Amazon helpfulness ([paper](https://arxiv.org/abs/2004.10964)) | Small, uses only training text (allowed by the brief). Covers the "domain mismatch" gap. Do not pretrain on test text (transductive) | Optional |
| **Transformer + TF-IDF stacking** | Works when base models are similarly strong | TF-IDF is 0.84 vs Ettin 0.947; it mostly repeats Ettin's easy wins | Skip (report as baseline only) |
| **Cascade to an LLM for low-confidence reviews** | Cascades cut cost with little accuracy loss; they cannot beat their best member by much ([2606.25871](https://arxiv.org/html/2606.25871)); fine-tuned small models beat GPT-4 by ~20 pts on in-domain labels | Reaches at most 23 of 44 errors. The LLM judges **text** sentiment, but our labels follow the **star-rating** convention — the label-fix experiment showed moving toward text sentiment *hurts* (0.947 → 0.930). Likely ≤ 0 | Skip as a submission method; fine as a (g) discussion point |
| **LLM-label distillation / pseudo-labelling test** | Gains reported on low-resource or fine-grained tasks | Same convention problem as above; pseudo-labelling test texts is a grey area under "train on training set only" | Skip |
| **Kaggle training tricks** (LLRD, AWP/FGM, multi-sample dropout, custom pooling) | AWP +0.9 pt in KDD Cup product classification ([paper](https://amazonkddcup.github.io/papers/1507.pdf)) | Below our measurement noise; can't be shown to help on 1,437 reviews | Skip |

Already decided against (not re-proposed): multiple seeds, model soups / weight averaging, threshold tuning, R-Drop,
chain-of-thought, LoRA on 7B LLMs, class weights for transformers, synthetic negatives, label cleaning.

Caveats on the ensemble:
- SiEBERT was fine-tuned on external sentiment data before we touch it. It is a pretrained checkpoint like Ettin, but
  ask the instructor whether that counts against "training must use only the training set".
- Fix the ensemble rule in advance (plain average). Choosing weights on validation inflates the validation score.

## 4. How benchmarks and industry get "very high" accuracy

| Benchmark | Top reported accuracy | How the labels are built |
|---|---|---|
| Yelp-2 (Yelp Polarity) | ~98.6% (XLNet-large) | 560k train reviews; star-derived |
| Amazon-2 (Amazon Polarity) | ~97.9% (XLNet-large) | 3.6M train reviews; 1–2★ negative, 4–5★ positive, **3★ removed** |
| IMDb | ~96–97% (XLNet; ERNIE-Doc) | ≤4/10 negative, ≥7/10 positive, **middle scores removed**; balanced |
| SST-2 | ~97% (DeBERTa-v3-large, T5-11B, ensembles) | Sentence-level, human-annotated |

Sources: [XLNet](https://arxiv.org/abs/1906.08237) (error 1.37 / 2.11 / 3.20 in the final version),
[Sun et al. 2019, How to Fine-Tune BERT](https://arxiv.org/abs/1905.05583), [GLUE leaderboard](https://gluebenchmark.com/leaderboard).

What drives those numbers:
1. **Ambiguous reviews are removed from the data.** Amazon Polarity and IMDb drop the middle ratings — exactly the mixed
   reviews that make up most of our errors. Our 97% on 5.7k reviews *with* mixed reviews is in the same league.
2. **Hundreds of thousands to millions of training reviews** (100–600× ours).
3. **Large encoders fine-tuned end to end** (XLNet, RoBERTa/DeBERTa-large), often after **in-domain pretraining**
   (BERT-ITPT-FiT, DAPT/TAPT).
4. **Ensembles** of different backbones on leaderboards and Kaggle; single models in production.

Industry practice (public evidence is thin; no company publishes its production model): fine-tuned transformer
encoders (RoBERTa/DeBERTa family) trained on star-derived labels are the norm for binary review sentiment; LLMs are
used for labelling, aspect extraction and summaries rather than as the binary classifier. Fine-tuned small encoders
beat prompted LLMs on stable labelled tasks ([Zhang 2023](https://arxiv.org/abs/2305.15005),
[Bucher & Martini 2024](https://arxiv.org/abs/2406.08660)).

## 5. Recommendation

1. Run DeBERTa-v3-large (already Stage 3) and a fine-tuned SiEBERT with the same protocol, saving `val_predictions.csv`.
2. Average the three models' probabilities; compare with McNemar against `ettin400m_final`, as for earlier changes.
3. Submit the ensemble only if it wins clearly; otherwise keep Ettin and report the ensemble as (g)/(h) material.
