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
normalizes them automatically (disable with `--no_normalize_losses`).

## Reproducing the results

```bash
# Investment: 4 betas over 2020-2026, ~1 minute
bash run_20_26_ru.sh            # optional arg: target alpha (default 0.01)

# Toxicity: 4 betas x 3 shift settings at T=10000
bash toxicity_scores/run_ru.sh
```

Both scripts resolve paths relative to themselves, so they work from any
directory and from a fresh clone. Requirements: `numpy`, `pandas`,
`matplotlib`, `scipy`, `tqdm`.

### Investment sweep

`run_20_26_ru.sh` pairs each CVaR level with the fixed weight whose risk matches
the target, and writes CSVs plus CVaR / rolling-CVaR / `c_t` / `λ_t` /
cumulative-return / loss plots into
`outputs_2020_2026_ru_alpha<α>_gamma0.05_noclip_normalized/`:

| β | `--baseline_lambda` |
| --- | --- |
| 0.90 | 0.36 |
| 0.85 | 0.46 |
| 0.80 | 0.54 |
| 0.75 | 0.62 |

The first 1,000 days are dropped as a tuning period (`--tune_days 1000`),
leaving a 496-day evaluation window (2024-01-04 to 2025-12-31).

> **Note.** This is the RU port of `run_20_26.sh`. Two flags from that script are
> intentionally gone: `--c0` and `--eps` do not exist in the RU algorithm
> (`c₁ = ½` and the AdaGrad-FTRL inner update replace them). `--lambda0` is also
> not passed, because `online_cvar_invest_ru.py`'s `main()` forwards
> `args.baseline_lambda` as `lambda0` to `run_one_setting()` — so `λ₁` always
> equals `--baseline_lambda` and the `--lambda0` flag has no effect.

### Toxicity sweep

`run_ru.sh` sweeps β ∈ {0.75, 0.8, 0.85, 0.9} across three distribution-shift
settings, where the sampling distribution over prompts is reweighted by a
`Beta(a_t, b)` density on the toxicity score:

- `uniform` — `a_t = 1` throughout.
- `adversarial` — `a_t` ramps piecewise-linearly 0.5 → 1 → 3, shifting mass
  toward toxic prompts over time.
- `adversarial_jump` — the same ramp, punctuated by 5 short randomly placed
  windows (all after `t = 1000`) that jump to `a ∈ {0.3, 5.0}`.

Each run also fits a static-λ baseline by distortion risk control over 1,000
candidate thresholds, for comparison against the adaptive λ.

To run a single configuration:

```bash
python toxicity_scores/online_cvar_detoxify_ru.py \
  --directory toxicity_scores/data_slim/llama3.2_real_toxic \
  --conformal_path toxicity_scores/data_slim/llama3.2_real_toxic/conformal_set_size_F1_0.26.pkl \
  --T 10000 --beta 0.9 --alpha 0.1 --gamma 0.05 --lambda0 1.0 \
  --beta_setting adversarial_jump --out_dir outputs_ru
```

## Data

The toxicity experiment uses Llama-3.2-3B generations on RealToxicityPrompts —
9,500 prompts × 40 responses, each scored by a fine-tuned Detoxify model
(machine score) and by the human-toxicity head, plus precomputed conformal sets.

The raw directory is 590 MB, and one file (`conformal_set_size_F1_0.26.pkl`,
172 MB) exceeds GitHub's hard 100 MB per-file limit. It is therefore hosted on
the Hugging Face Hub:

> **[`Evangelinejy/online-cvar-llama3.2-real-toxic`](https://huggingface.co/datasets/Evangelinejy/online-cvar-llama3.2-real-toxic)**

```bash
bash toxicity_scores/download_raw_data.sh
DATA_ROOT=toxicity_scores/data/llama3.2_real_toxic bash toxicity_scores/run_ru.sh
```

**You do not need the raw download.** `online_cvar_detoxify_ru.py` reads only
four things from those pickles:

- `detoxify_ft` — machine scores
- `detoxify_human["toxicity"]` — human scores
- the response *indices* in `conformal[key]["set"]` (the texts there are
  discarded)
- `pred`, which is loaded but never read afterwards

The generated response texts and perplexities — ~99% of the bytes — are never
used. `data_slim/` keeps only the fields above, in the identical nested
structure, so the script runs against it unchanged. This was verified two ways:

1. **Exhaustively**, over all 9,500 prompts: 380,000 machine scores and 380,000
   human scores compare bit-for-bit equal, and every conformal index list
   matches.
2. **End to end**: the same configuration run against `data_slim/` and against
   the raw data produced byte-identical output CSVs, the same static λ
   (0.011900), and the same final realized CVaR (0.2612238774696986).

Regenerate the slim copy with:

```bash
python toxicity_scores/build_slim_data.py \
  --src toxicity_scores/data/llama3.2_real_toxic \
  --dst toxicity_scores/data_slim/llama3.2_real_toxic
```

A side benefit is that no toxic generated text is published to this repository.

> `generated_responses_999.pkl` is a 48-byte empty pickle in the source data and
> contributes no prompts; it is kept so the file listing matches the raw
> directory.
