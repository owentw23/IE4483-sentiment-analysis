---
tags: [resources, lecture]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: sources below; Ettin shapes from docs/report_draft_abc.md
---
# L4 — Classifier architectures (question b)

Part of [[MOC - ML Training Lecture]]. Previous: [[L3 - Turning text into features]].

Every classifier in this project has the same skeleton: **encoder** (text → one vector per review) → **head** (vector → 2 logits) → **softmax** → **cross-entropy**. Only the encoder changes. Question (b) wants this drawn with **tensor shapes**, written as `[B, L, d]` = [batch size, sequence length in tokens, vector width].

## 1. Logistic regression on TF-IDF: the perceptron, upgraded

```
TF-IDF vector [B, 14114] → Linear 14114→2 → softmax → [B, 2]
```
This is L1's perceptron with the step function replaced by softmax (smooth, so it has gradients) and squared error replaced by CE. It has no hidden layer, so the boundary is linear in TF-IDF space. scikit-learn fits it with a dedicated solver rather than SGD, but the objective is the same: (class-weighted) cross-entropy plus an L2 penalty (`C` = inverse regularisation strength).

## 2. BiLSTM + GloVe: the RNN baseline

```
token ids [B, L]
 → GloVe embedding lookup          [B, L, 300]   (frozen or fine-tuned)
 → BiLSTM, hidden 256 per direction [B, L, 512]   (forward ‖ backward)
 → max-pool over time               [B, 512]      (strongest signal per feature)
 → dropout → Linear 512→2           [B, 2]
```
- **Pooling** turns L vectors into one. Max-pooling asks "did any position fire this feature?", which suits sentiment cues that can appear anywhere. The alternatives are the final hidden state or the mean.
- **Dropout** randomly zeroes activations during training, a regulariser (L5).
- This is the "RNN + linear layer" example the brief itself names.

## 3. Transformer encoder + head: the main model (Ettin-400M)

```
token ids, attention mask           [B, L]        (L ≤ 512)
 → token embedding                  [B, L, 1024]
 → 28 transformer layers            [B, L, 1024]  (self-attention + GeGLU FFN, pre-norm,
                                                   RoPE; local 128-token attention,
                                                   global every 3rd layer)
 → masked MEAN pooling              [B, 1024]     (average of real tokens only)
 → head: Linear 1024→1024 + GELU + LayerNorm
 → Linear 1024→2                    [B, 2]  → softmax → P(neg), P(pos)
```
Key points:
- **Pooling choice.** BERT-style models classically use the `[CLS]` token's vector. Ettin/ModernBERT default to **mean pooling** (`classifier_pooling: mean`), so every token contributes. Either works once fine-tuned; state which one you use.
- **The head is randomly initialised**; the encoder starts from pretrained weights. That mismatch is why warm-up matters (L5): early on, the random head's large gradients flow into the pretrained layers.
- In Hugging Face this whole stack is `AutoModelForSequenceClassification.from_pretrained("jhu-clsp/ettin-encoder-400m", num_labels=2)`. The library adds the head for you and warns that the head weights are "newly initialized". That warning is expected.
- DeBERTa-v3-large (the baseline to beat) has the same skeleton: 24 layers, 1024-d, a different attention variant, and `[CLS]`-style pooling.

## 4. Decoder (GPT-style) as a classifier

A causal decoder only sees *leftward* context, so the **last real token** is the only one that has seen the whole review. Use its hidden state and a linear head to get 2 logits (`AutoModelForSequenceClassification` does this for decoder models). The Ettin decoder twin is this, with identical data and size to the encoder. The comparison isolates the *training objective* (masked LM vs next-token): a clean encoder-vs-decoder experiment. The Ettin paper reports that encoders win on classification at equal size.

## 5. Zero-shot classifiers (for question g, no training)

- **Already-fine-tuned sentiment model** (SiEBERT: RoBERTa-large trained on 15 sentiment datasets). Just run it.
- **NLI-based zero-shot.** Ask an entailment model whether the review entails "This review is positive."
- **GLiClass.** Label names and text go into one encoder pass, and each label is scored.
- **Decision models / prompted LLMs.** Ask a generative model to pick a label.

These need no labels, but they can't adapt to dataset quirks such as our star-rating-derived labels.

## 6. What a strong (b) answer contains

1. An architecture **figure** with shapes at every stage (like §3).
2. Input and output definitions: what one sample is, what the output means, how a label is predicted (argmax).
3. Parameter count (about 400M) and which parts are pretrained vs new.
4. **Loss**: class-weighted CE, with the formula and the weights.
5. **Training strategy**: optimizer, learning-rate schedule, batch size, epochs, checkpoint selection, precision and memory tricks (L5, L6).
6. **Code and how to run it**: the script name, the exact commands, required packages.
7. **Reproducibility**: fixed seeds, a deterministic split, pinned versions. The brief explicitly accepts "fixing random seed".

> [!question]- Check yourself 1: why does a decoder classifier use the last token while an encoder can use the first ([CLS]) token or the mean?
> In a causal decoder, token *i* only attends to tokens ≤ *i*, so only the last token has seen everything. In a bidirectional encoder every token sees the whole review, so any pooling works.

> [!question]- Check yourself 2: in the Ettin stack, which weights are pretrained and which are random at the start of fine-tuning?
> Embeddings and all 28 transformer layers are pretrained. The classification head (the dense layer and the final 1024→2 linear) is newly initialised.

> [!question]- Check yourself 3: write the output shape after each step for a batch of 8 reviews padded to 120 tokens.
> ids/mask [8, 120] → embeddings [8, 120, 1024] → encoder [8, 120, 1024] → mean-pool [8, 1024] → head [8, 1024] → logits [8, 2].

## Sources and further study

- Hugging Face docs, text classification task guide: https://huggingface.co/docs/transformers/tasks/sequence_classification
- Weller et al. 2025, *Seq vs Seq* (Ettin encoders vs decoders): https://arxiv.org/abs/2507.11412
- Answer.AI / HF, *Introducing ModernBERT* (the architecture Ettin shares): https://huggingface.co/blog/modernbert
- Olah, *Understanding LSTM Networks*: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- HF blog, zero-shot classification with NLI: https://huggingface.co/blog/zero-shot-classification
