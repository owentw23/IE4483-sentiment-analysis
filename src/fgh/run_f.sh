#!/usr/bin/env bash
# Report (f): every feature-format arm, one process per arm (so peak memory belongs to that arm), on the
# GPU server only. Reuses the repo's infrastructure: split from src/data.py, per-arm folders in outputs/runs/
# (same layout as src/train.py, so src/evaluate.py works on them), f11 trained by src/train.py itself.
# Results: outputs/runs/<arm>/, ledger outputs/results_f.csv; Table F1 + figures in outputs/report_f/.
#
# One-time setup on the server (repo cloned with Git LFS so the final model weights are real):
#   git lfs install && git lfs pull
#   pip install uv && uv venv -p 3.12 .venv && source .venv/bin/activate
#   uv pip install -r requirements.txt && uv pip install scipy tabulate
# Run:  bash src/fgh/run_f.sh 2>&1 | tee outputs/fgh_f_run.log
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}

for arm in $($PY src/fgh/tfidf_arms.py --list); do $PY src/fgh/tfidf_arms.py --arm "$arm"; done   # f1-f3
for d in 50 100 300; do $PY src/fgh/glove_arms.py --dim "$d"; done                                 # f4
$PY src/fgh/bilstm.py --arm f5_bilstm_glove300_frozen                                               # f5
$PY src/fgh/bilstm.py --arm f5b_bilstm_scratch                                                      # f5b
for arm in f6_qwen3_embedding_frozen f7_ettin400m_frozen f10_siebert; do                            # f6 f7 f10
  $PY src/fgh/pretrained_arms.py --arm "$arm"
done
$PY src/fgh/ettin_maxlen.py                                                                         # f9
$PY src/train.py --run-name ettin150m_lr2e-5 --model jhu-clsp/ettin-encoder-150m                    # f11
$PY src/fgh/report_f.py                                                                             # table + figures
