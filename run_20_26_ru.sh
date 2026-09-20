#!/bin/bash
# RU Conformal Inference port of run_20_26.sh: S&P 500 (^GSPC) vs 10-year
# Treasury yield (DGS10) over 2020-2026, sweeping the four CVaR levels.
#
# Differences from run_20_26.sh, which targets online_cvar_invest.py:
#   * --c0 and --eps are gone. The RU algorithm fixes c_1 = 1/2 and adapts c
#     through the AdaGrad--FTRL inner update, so neither flag exists.
#   * --lambda0 is NOT passed. online_cvar_invest_ru.py's main() hands
#     baseline_lambda to run_one_setting() as lambda0, so lambda_1 always
#     equals --baseline_lambda and the --lambda0 flag has no effect.
#
# Each beta keeps the baseline_lambda it was paired with in run_20_26.sh (the
# fixed weight whose CVaR matches the target at that level).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_CSV="2020_2026.csv"
CSV_STEM="${DATA_CSV%.csv}"
ALPHA=${1:-0.01}
GAMMA=0.05

BETAS=(0.90 0.85 0.80 0.75)
BASELINE_LAMBDAS=(0.36 0.46 0.54 0.62)

OUT_DIR="outputs_${CSV_STEM}_ru_alpha${ALPHA}_gamma${GAMMA}_noclip_normalized"

for i in "${!BETAS[@]}"; do
  beta="${BETAS[$i]}"
  baseline_lambda="${BASELINE_LAMBDAS[$i]}"

  echo "=========================================="
  echo "Running RU invest: beta=${beta}, baseline_lambda=${baseline_lambda}, alpha=${ALPHA}, gamma=${GAMMA}"
  echo "Output: ${OUT_DIR}"
  echo "=========================================="

  python online_cvar_invest_ru.py \
    --data_csv "${DATA_CSV}" \
    --date_col date --risky_col '^GSPC' --rf_col DGS10 \
    --horizon 1 \
    --rf_is_yield --periods_per_year 252 \
    --beta "${beta}" \
    --alpha "${ALPHA}" \
    --gamma "${GAMMA}" \
    --baseline_lambda "${baseline_lambda}" \
    --tune_days 1000 \
    --out_dir "${OUT_DIR}"

  echo ""
done

echo "All ${#BETAS[@]} betas completed! Results under ${OUT_DIR}"
