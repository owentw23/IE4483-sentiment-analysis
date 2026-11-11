#!/usr/bin/env bash
# Report (f): every feature-format arm, one process per arm (so peak memory belongs to that arm), on the
# GPU server only. Reuses the repo's infrastructure: split from src/data.py, per-arm folders in outputs/runs/
# (same layout as src/train.py, so src/evaluate.py works on them), f11 trained by src/train.py itself.
# Results: outputs/runs/<arm>/, ledger outputs/results_f.csv; Table F1 + figures in outputs/report_f/.
# A failing step does not stop the run: every step is attempted and the failures are listed at the end.
#
# One-time setup on the server (repo cloned with Git LFS so the final model weights are real):
#   git lfs install && git lfs pull
#   uv venv -p 3.12 .venv && source .venv/bin/activate
#   uv pip install -r requirements.txt && uv pip install scipy tabulate
# Run:  bash src/fgh/run_f.sh 2>&1 | tee outputs/fgh_f_run.log
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
FAILED=()

run() {  # run one step; on failure, remember it and carry on
  echo
  echo "===== $* ====="
  if ! "$@"; then
    FAILED+=("$*")
    echo "!!!!! FAILED: $*"
  fi
}

for arm in $($PY src/fgh/tfidf_arms.py --list); do run $PY src/fgh/tfidf_arms.py --arm "$arm"; done  # f1-f3
for d in 50 100 300; do run $PY src/fgh/glove_arms.py --dim "$d"; done                                # f4
run $PY src/fgh/bilstm.py --arm f5_bilstm_glove300_frozen                                              # f5
run $PY src/fgh/bilstm.py --arm f5b_bilstm_scratch                                                     # f5b
for arm in f6_qwen3_embedding_frozen f7_ettin400m_frozen f10_siebert; do                               # f6 f7 f10
  run $PY src/fgh/pretrained_arms.py --arm "$arm"
done
run $PY src/fgh/ettin_maxlen.py                                                                        # f9
run $PY src/train.py --run-name ettin150m_lr2e-5 --model jhu-clsp/ettin-encoder-150m                   # f11
run $PY src/train.py --run-name ettin400m_final_3090 --save-model --predict-test                       # f8b: final recipe on this GPU
run $PY src/fgh/ettin_maxlen.py --run ettin400m_final_3090 --arm-prefix f9_ettin3090                   # f9 on the f8b weights
run $PY src/fgh/report_f.py                                                                            # table + figures

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "ALL STEPS OK"
else
  echo "FAILED STEPS (${#FAILED[@]}):"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
