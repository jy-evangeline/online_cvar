# Online CVaR Control (Rockafellar–Uryasev)

Online control of the **Conditional Value-at-Risk (CVaR)** of a loss sequence via the
Rockafellar–Uryasev (RU) variational representation, applied to two settings:
risk-managed portfolio allocation and toxicity filtering for LLM outputs.

Both implementations share the same two-level scheme:

- **Outer level** — a coordinate-descent-threshold (CDT) update on the decision
  variable `λ`:  `λ_{t+1} = λ_t − γ (ĈVaR_t − α)`.
- **Inner level** — an AdaGrad-FTRL update on the RU auxiliary variable
  `c ∈ [0, 1]`, whose regularized leader

  ```
  (1 / 2η) (c − ½)² + Σ_{s≤t} [ c + (1−β)⁻¹ (R_s − c)₊ ]
  ```

  is convex with a non-decreasing derivative, so its exact minimizer is found by
  bisection on the zero crossing (`_ftrl_argmin_c`).

The RU surrogate estimate at step `t` is `ĈVaR_t = c_t + (1−β)⁻¹ (L_t − c_t)₊`.
Initialization is `c₁ = ½`, `q₀ = max(1, β/(1−β))²`. Because `c` is constrained to
`[0, 1]`, losses are expected on roughly that scale — the investment script
normalizes them automatically (see `--no_normalize_losses`).

## Contents

| File | Description |
| --- | --- |
| `online_cvar_invest_ru.py` | Portfolio allocation. `λ_t` is the risky-asset weight; loss is `L_t(λ) = −(λ·r_risky + (1−λ)·r_rf)`, extended outside `[lam_min, lam_max]` by its inf/sup over the range (the loss is linear in `λ`, so both are attained at the endpoints). Emits a per-period CSV plus CVaR / rolling-CVaR / `c_t` / `λ_t` / cumulative-return / loss plots. |
| `toxicity_scores/online_cvar_detoxify_ru.py` | LLM toxicity control. `λ_t` is a threshold on machine (Detoxify fine-tuned) scores; the loss is the human toxicity of the responses admitted at that threshold. Machine scores are rank-normalized to an empirical CDF on `(0, 1]`. Supports `uniform`, `adversarial` (piecewise-linear ramp 0.5 → 1 → 3) and `adversarial_jump` distribution shift, and compares against a static-λ distortion-risk-control baseline. |
| `toxicity_scores/run_ru.sh` | Sweep driver for the toxicity experiment: 4 CVaR levels β ∈ {0.75, 0.8, 0.85, 0.9} × 3 shift settings. |

## Usage

### Portfolio allocation

```bash
python online_cvar_invest_ru.py \
  --data_csv 1987_2026.csv \
  --risky_col risky_close --rf_col rf_yield --rf_is_yield \
  --beta 0.9 --alpha 0.01 --gamma 0.05 \
  --baseline_lambda 0.5 \
  --no_normalize_losses \
  --out_dir outputs_1987_2026_ru
```

The input CSV needs a date column plus a risky-asset close and a risk-free column
(a price by default, or an annualized yield in percent with `--rf_is_yield`).
`--baseline_lambda` sets both `λ₁` and the fixed-weight comparison portfolio.
Period filtering is available via `--year` / `--quarter` / `--months`, and
`--tune_days` drops leading days from both the online method and the baseline.

### Toxicity control

```bash
bash toxicity_scores/run_ru.sh
```

Edit the paths at the top of the script first: `DIRECTORY` points at a folder of
`.pkl` files whose entries carry `detoxify_ft` (machine score) and
`detoxify_human.toxicity` (human score) per response, and `CONFORMAL_PATH` at the
pickled conformal sets used to fit the static-λ baseline. Both currently default
to absolute `/scratch` paths.

To run a single configuration directly:

```bash
python toxicity_scores/online_cvar_detoxify_ru.py \
  --directory <pkl_dir> --conformal_path <conformal.pkl> \
  --T 10000 --beta 0.9 --alpha 0.1 --gamma 0.05 --lambda0 1.0 \
  --beta_setting adversarial_jump --out_dir outputs_ru
```

Note that the RU scripts take no `--eps`, `--c0`, `--burn_in` or `--truncated`
flags — `c₁ = ½` and the AdaGrad-FTRL inner update replace them.

## Requirements

`numpy`, `pandas`, `matplotlib`, `scipy`, `tqdm`.
