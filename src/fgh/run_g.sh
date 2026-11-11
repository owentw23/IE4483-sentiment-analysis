#!/usr/bin/env bash
# Report (g): the hypothetical hotel project, simulated with TripAdvisor reviews whose stars are hidden from every
# method. GPU server only. Same conventions as run_f.sh: one process per step, failures listed at the end.
#
#   bash src/fgh/run_g.sh prep    # G1-G5: download, clean, answer key, user-grouped splits, shift diagnostics
#   bash src/fgh/run_g.sh check   # G6: every model loads and labels 10 product reviews (writes nothing)
#   bash src/fgh/run_g.sh arms    # g0-g4c: methods that need no hotel labels and no training
#   bash src/fgh/run_g.sh all     # prep, check, arms
# Tier C (D1 = C), run after `arms` (needs the g1/g2 pool scores):
#   bash src/fgh/run_g.sh selftrain   # g5, g5b: pseudo-labels -> fresh Ettin-400M student via src/train.py
#   bash src/fgh/run_g.sh dapt        # g6: masked-LM on hotel pool text -> fine-tune via src/train.py
#   bash src/fgh/run_g.sh fewshot     # g7a (Ettin again), g7b (SetFit; needs `uv pip install setfit`)
# Student checkpoints go to data/external/models/ (git-ignored), not outputs/.
# Log: bash src/fgh/run_g.sh all 2>&1 | tee outputs/fgh_g_run.log
# Extra packages (once): uv pip install -p .venv/bin/python pyarrow gliclass
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
STAGE=${1:-all}
FAILED=()

run() {  # run one step; on failure, remember it and carry on
  echo
  echo "===== $* ====="
  if ! "$@"; then
    FAILED+=("$*")
    echo "!!!!! FAILED: $*"
    return 1
  fi
}
NOTRAIN=(g0_tfidf_products g1_ettin_final g1t_ettin_final_title g2_finecat_nli_v2 g2s_ettin68m_nli
         g3_deberta_zeroshot_c g4_gliclass_modern_large g4b_siebert g4c_decision_kai)

if [[ $STAGE == prep || $STAGE == all ]]; then
  run $PY src/fgh/hotel_data.py                                                       # G1-G4
  run $PY src/fgh/hotel_shift.py                                                      # G5
fi
if [[ $STAGE == check || $STAGE == all ]]; then
  for arm in "${NOTRAIN[@]}"; do run $PY src/fgh/hotel_arms.py --arm "$arm" --check; done  # G6
fi
if [[ $STAGE == arms || $STAGE == all ]]; then
  run $PY src/fgh/hotel_arms.py --arm g1_ettin_final --pool      # first: the hotel McNemar reference; g5 teacher
  run $PY src/fgh/hotel_arms.py --arm g2_finecat_nli_v2 --pool   # pool scores feed the g5b agreement filter
  run $PY src/fgh/hotel_arms.py --arm g2s_ettin68m_nli --pool
  for arm in g0_tfidf_products g1t_ettin_final_title g3_deberta_zeroshot_c g4_gliclass_modern_large \
             g4b_siebert g4c_decision_kai; do
    run $PY src/fgh/hotel_arms.py --arm "$arm"
  done
fi

MODELS=data/external/models
student() {  # student <arm> <train.py args...>: train with the submitted recipe, park the checkpoint, score it
  local arm=$1; shift
  run $PY src/train.py --run-name "${arm}_model" --save-model "$@" || return
  if [ -d "outputs/runs/${arm}_model/model" ]; then
    mkdir -p "$MODELS" && rm -rf "$MODELS/$arm" && mv "outputs/runs/${arm}_model/model" "$MODELS/$arm"
    run $PY src/fgh/hotel_arms.py --arm "$arm"
  fi
}
if [[ $STAGE == selftrain || $STAGE == selftrain_g5 ]]; then
  run $PY src/fgh/self_train.py --variant g5
  student g5_selftrain --extra-train data/external/hotel/pseudo_g5.csv
fi
if [[ $STAGE == selftrain || $STAGE == selftrain_g5b ]]; then
  run $PY src/fgh/self_train.py --variant g5b
  student g5b_selftrain_agree --extra-train data/external/hotel/pseudo_g5b.csv
fi
if [[ $STAGE == dapt ]]; then
  run $PY src/fgh/dapt.py
  student g6_dapt --model "$MODELS/g6_dapt_mlm"
fi
if [[ $STAGE == fewshot ]]; then
  run $PY src/fgh/fewshot.py --method ettin --k 8 16 25
fi
if [[ $STAGE == fewshot || $STAGE == setfit ]]; then
  run $PY src/fgh/fewshot.py --method setfit --k 8 16 25
fi

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "ALL STEPS OK ($STAGE)"
else
  echo "FAILED STEPS (${#FAILED[@]}):"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
