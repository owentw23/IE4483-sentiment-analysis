---
tags: [resources, lecture]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: sources below; project numbers from CLAUDE.md and docs/report_draft_abc.md
---
# L3 — Turning text into features (questions a and f)

Part of [[MOC - ML Training Lecture]]. Previous: [[L2 - Data splits, metrics and class imbalance]].

Networks take numbers, not text. A **feature format** is the mapping from a review to numbers. The brief names three families; they get richer and more expensive in that order.

## 1. Bag-of-words and TF-IDF (sparse, counted)

1. **Tokenise into words** (lowercase, split on punctuation). Optionally add **n-grams**: bigrams like "not good" partly recover word order.
2. **Vocabulary**: every distinct term in the *training* split. Ours has about 14k terms with 1–2-grams.
3. Each review becomes a vector of length |vocabulary|, almost all zeros (*sparse*).
4. Weight each entry with TF-IDF:
$$\text{tfidf}(t,d) = \text{tf}(t,d)\times\text{idf}(t),\qquad \text{idf}(t)=\ln\frac{1+N}{1+\text{df}(t)}+1$$
   (scikit-learn's smoothed form, followed by L2-normalising each row). $\text{tf}$ = how often term $t$ appears in review $d$; $\text{df}$ = how many reviews contain $t$. Common words like "the" get low IDF; rare, informative ones like "refund" get high IDF.

**Strengths:** fast (seconds on CPU), easy to interpret (each weight belongs to a word), strong on short, keyword-driven text.

**Weaknesses:**
- **No meaning.** "awful" and "terrible" are unrelated columns.
- **No context.** Order is lost beyond n-grams.
- **Unseen words** are ignored.
- **Negation and contrast** ("looks great *but* broke after a week") are handled poorly.

Pair TF-IDF with a linear classifier (logistic regression). That model is a single-layer perceptron with a sigmoid output and cross-entropy loss: exactly L1's math, one weight per word.

## 2. Static word embeddings: word2vec and GloVe (dense, pretrained)

- Each word becomes a **dense** vector, for example 300 numbers, learned beforehand from billions of words so that words used in similar contexts get similar vectors. word2vec predicts neighbouring words; GloVe factorises global co-occurrence statistics.
- "awful" ≈ "terrible" in vector space, so meaning **transfers** from huge unlabelled corpora to our small dataset.
- **Static**: "good" has *one* vector everywhere. "not good" is just "not" next to "good".
- **How to get one vector per review**:
  - Simplest: **average** the word vectors. Cheap, but order is lost again.
  - Better: feed the *sequence* of vectors into an **RNN/LSTM**, which reads left to right while carrying a hidden state, so it *can* learn that "not" flips "good".
  - **BiLSTM**: also read right to left and concatenate both directions. That's the project's BiLSTM + GloVe arm.
- **LSTM in one line**: gates (forget, input, output) control what goes into a memory cell. The cell's additive update lets gradients flow through long sequences without vanishing, which plain RNNs suffer from (L1's saturation problem, repeated over many time steps).

## 3. Subword tokenisation (what transformers read)

Transformers don't use a fixed word list. **BPE/WordPiece** tokenisers split text into frequent subword pieces, so any string is representable and no word is unknown: "unboxing" becomes "un" + "box" + "ing". Ettin's vocabulary has 50,368 pieces.

Example from our report draft: `"Not good, returned it!!"` → `[CLS] Not Ġgood , Ġreturned Ġit !! [SEP]`, where Ġ marks a leading space. The tokenizer is **pretrained and fixed**, so nothing is fitted on our data and there's no leakage risk.

**Practical knobs:**
- **max_length / truncation**: cut long reviews. We use 512 tokens, which covers 99.67% of reviews whole.
- **Padding**: batches need equal lengths. **Dynamic padding** pads each batch only to its own longest review, with an **attention mask** (1 = real token, 0 = pad) telling the model to ignore padding.
- **Cleaning**: for transformers, do **minimal cleaning**. Keep case ("TERRIBLE"), stop-words ("not", "but") and punctuation ("!!"); the pretrained model learned to use them. Heavy cleaning (lowercasing, stemming, stop-word removal) is a TF-IDF-era habit and *hurts* transformers.

## 4. Contextual embeddings (pretrained language models)

You already know transformers, so here's what matters for classification:
- Each token's output vector depends on the **whole review** through self-attention, so "good" in "not good" gets a different vector from "good" alone. That's *contextual*.
- Pretraining on trillions of tokens (Ettin: about 2T) gives the model broad language knowledge before it sees a single review. This matters a lot when you have only 5.7k labelled examples.
- **Two ways to use them:**
  1. **Frozen features (feature extraction).** Run the pretrained model once, take one vector per review (an embedding model like Qwen3-Embedding-0.6B gives 1024 numbers), then fit logistic regression on top. Cheap: one forward pass per review and no backprop through the big model.
  2. **Fine-tuning.** Put a small classification head on top and **train all weights end to end** with our labels. The representation itself adapts to sentiment. Usually the most accurate, and by far the most expensive (L6). This is the project's main approach.

## 5. Resource vs accuracy trade-off (the core of question f)

| Feature format (arm) | Trainable params | Hardware and train time (expected) | Captures | Validation macro-F1 |
|---|---|---|---|---|
| TF-IDF 1–2 gram + LogReg | ~14k | CPU, seconds | Keywords, short phrases | 0.841 (measured, class-weighted) |
| GloVe 300-d + BiLSTM | ~1M (LSTM; embeddings often frozen) | GPU or CPU, minutes | Meaning plus order, learned from 5.7k reviews | to measure |
| Frozen Qwen3 embeddings + LogReg | ~1k (head only; 0.6B frozen) | GPU, minutes (one forward pass per review) | Rich general semantics, not task-tuned | to measure |
| Zero-shot SiEBERT (RoBERTa-large, already sentiment-tuned) | 0 (no training) | GPU, inference only | Generic sentiment | 0.921 (measured) |
| Ettin-400M fine-tuned | ~400M | 8 GB GPU with memory tricks, tens of minutes to hours | Everything, adapted to this data | to measure (projected 0.92–0.95) |

**How to write (f).** Accuracy rises down the table, but compute, memory, energy and latency rise by orders of magnitude. The answer should:
1. Quantify both axes with *your measured numbers*: parameters, peak GPU memory, training time, inference time, macro-F1.
2. Interpret the curve: where the big jump comes from (meaning and context, pretraining knowledge) and where returns diminish.
3. Note when a cheaper format is the right engineering choice: a CPU-only deployment, millions of reviews per day, or a need for explainable keyword weights.

That's why Stage 3 must log time and memory, not just scores.

> [!question]- Check yourself 1: why might bigrams help TF-IDF but still fail on "I expected to hate it, but it's great"?
> Bigrams capture adjacent pairs like "not good", but this sentence's sentiment depends on a long-range contrast ("expected to hate … but … great"). Only sequence-aware models (LSTM) or contextual ones (transformers) can model it.

> [!question]- Check yourself 2: why shouldn't you lowercase and remove stop-words before a transformer?
> The tokenizer and model were pretrained on natural text. Case and stop-words carry signal ("not", "but", "TERRIBLE"), and removing them creates input unlike anything seen in pretraining.

> [!question]- Check yourself 3: what's the difference between frozen features and fine-tuning, in terms of what gets updated by backprop?
> Frozen: only the small head (for example logistic regression) is trained, and the big model's weights never change. Fine-tuning: gradients flow through all layers and every weight is updated.

## Sources and further study

- scikit-learn, text feature extraction (bag-of-words, TF-IDF): https://scikit-learn.org/stable/modules/feature_extraction.html
- Alammar, *The Illustrated Word2vec*: https://jalammar.github.io/illustrated-word2vec/
- Stanford GloVe project and vectors: https://nlp.stanford.edu/projects/glove/
- Olah, *Understanding LSTM Networks*: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Karpathy, *The Unreasonable Effectiveness of RNNs*: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Hugging Face LLM Course, tokenizers: https://huggingface.co/learn/llm-course/en/chapter2/4
- Alammar, *The Illustrated BERT*: https://jalammar.github.io/illustrated-bert/
- Course notebook: `tfidf_features_example.ipynb` (repo root)
