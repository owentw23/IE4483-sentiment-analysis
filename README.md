# IE4483 Project 1 — Sentiment Analysis of Product Reviews

Binary sentiment classifier (0 = negative, 1 = positive) for product reviews.

## Data
- `train.json` — 7,401 labelled reviews (`reviews`, `sentiments`); ~85% positive
- `test.json` — 1,851 unlabelled reviews → predictions go to `submission.csv`
- `instruction1.png`, `instruction2.png` — project brief
- `tfidf_features_example.ipynb` — course-provided TF-IDF example

## Planned models
- Baselines: TF-IDF + Logistic Regression, BiLSTM + GloVe, SiEBERT (zero-shot)
- Main: `jhu-clsp/ettin-encoder-400m` (fallback `ettin-encoder-150m`), vs `microsoft/deberta-v3-large`
- Techniques: class-weighted loss, LLM-generated synthetic negatives (ablation)

## Setup (Windows + NVIDIA RTX 50-series)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers scikit-learn pandas
```

## Optional: Colab MCP (local Claude Code only)
```bash
claude mcp add --scope project colab-mcp -- uvx git+https://github.com/googlecolab/colab-mcp
```
