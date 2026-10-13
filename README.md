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

## Setup (Windows, NVIDIA GPU)
```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Run
```bash
.venv/Scripts/python src/data.py                      # check the split and class weights
.venv/Scripts/python src/sanity_check.py              # initial loss ~0.69 + overfit one batch
.venv/Scripts/python src/train.py --run-name NAME     # one run (defaults = planned config)
.venv/Scripts/python src/sweep.py --tag lr --lr 1e-5 2e-5 3e-5   # grid search
.venv/Scripts/python src/sweep.py --show              # results ledger, best first
.venv/Scripts/python src/predict.py --model-dir outputs/runs/NAME/model   # submission.csv
```
Details: `docs/training_guide.md`.

## Optional: Colab MCP (local Claude Code only)
```bash
claude mcp add --scope project colab-mcp -- uvx git+https://github.com/googlecolab/colab-mcp
```
