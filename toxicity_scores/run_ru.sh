#!/bin/bash
# Run the RU Conformal Inference variant (online_cvar_detoxify_ru.py) for the
# three beta settings: uniform, adversarial, adversarial_jump.
#
# Note: the RU script has NO --eps / --c0 / --burn_in / --truncated flags
# (c_1=1/2 and the AdaGrad--FTRL update replace them).
set -euo pipefail

# Base paths. DATA_ROOT defaults to the slim (numeric-only) copy committed to
# this repo, so a fresh clone runs without downloading anything. To run against
# the raw generations instead -- identical results, see README -- override it:
#   DATA_ROOT=/path/to/llama3.2_real_toxic bash run_ru.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data_slim/llama3.2_real_toxic}"
DIRECTORY="${DIRECTORY:-${DATA_ROOT}}"
CONFORMAL_PATH="${CONFORMAL_PATH:-${DATA_ROOT}/conformal_set_size_F1_0.26.pkl}"
MODEL_NAME="Llama3.2-3B"

# Fixed parameters (edit as needed)
T=10000
LAMBDA0=1.0          # initial lambda_1 (clamped to [lam_min,lam_max] only for loss eval)
BETAS=(0.75 0.8 0.85 0.9)   # CVaR levels to sweep
ALPHA=0.1            # target risk level
GAMMA=0.05           # outer (CDT) step size
CAL_SIZE=500
TEST_SIZE=1000

# adversarial_jump knobs
N_JUMPS=5
JUMP_ALPHAS="0.3,5.0,0.3,5.0,0.3"
JUMP_SEED=42

RESULTS_ROOT="./results_ru_ct_alpha${ALPHA}_${MODEL_NAME}_lambda${LAMBDA0}_gamma${GAMMA}"

BETA_SETTINGS=("uniform" "adversarial" "adversarial_jump")

for beta in "${BETAS[@]}"; do
  for beta_setting in "${BETA_SETTINGS[@]}"; do
    OUT_DIR="${RESULTS_ROOT}/outputs_online_detoxify_${beta_setting}_beta${beta}"

    echo "=========================================="
    echo "Running RU: beta_setting=${beta_setting}, beta=${beta}, gamma=${GAMMA}, alpha=${ALPHA}, T=${T}"
    echo "Output: ${OUT_DIR}"
    echo "=========================================="

    python "${SCRIPT_DIR}/online_cvar_detoxify_ru.py" \
      --directory "${DIRECTORY}" \
      --conformal_path "${CONFORMAL_PATH}" \
      --T "${T}" \
      --beta_setting "${beta_setting}" \
      --beta "${beta}" \
      --alpha "${ALPHA}" \
      --gamma "${GAMMA}" \
      --lambda0 "${LAMBDA0}" \
      --cal_size "${CAL_SIZE}" \
      --test_size "${TEST_SIZE}" \
      --n_jumps "${N_JUMPS}" \
      --jump_alphas "${JUMP_ALPHAS}" \
      --jump_seed "${JUMP_SEED}" \
      --out_dir "${OUT_DIR}"

    echo ""
  done
done

echo "All sweeps completed! (${#BETAS[@]} betas x ${#BETA_SETTINGS[@]} settings) Results under ${RESULTS_ROOT}"
