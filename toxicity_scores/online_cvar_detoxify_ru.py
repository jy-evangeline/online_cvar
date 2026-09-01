import os
import bisect
import argparse
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.ticker import MultipleLocator
from scipy.stats import rankdata
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Plotting style ported from the plots notebook: large bold fonts, dark grid,
# fixed palette, no titles. PLOT_BURN_IN drops the first steps from PLOTS ONLY
# (the saved CSV keeps every step).
# ----------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
FIGSIZE = (13, 7)
LABEL_SIZE = 30
LEGEND_SIZE = 18
TICK_SIZE = 25
LINE_WIDTH = 3
PLOT_DPI = 300
PLOT_BURN_IN = 100
AX_FACE = "#EAEAF2"
FIG_FACE = "white"
COLORS = {
    "portfolio": "#4C97BF",
    "risky": "#d62728",
    "rf": "#2ca02c",
    "baseline": "#8C4FAF",
    "target": "#D62728",
    "full": "#F4A024",
    "primary": "#4C97BF",
    "secondary": "#8C4FAF",
}


def _style_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.figure.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(AX_FACE)
    ax.set_title("")
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.35)


def _finalize(fig, ax, path: str) -> None:
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_SIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def resample_beta_weighted(
    x: np.ndarray,
    alpha: float,
    beta: float,
    n_samples: int,
    replace: bool = True,
    eps: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
    return_probs: bool = False,
):
    """
    Weighted resampling from discrete samples x in [0,1] using Beta(alpha, beta) pdf as weights.

    Args:
        x: array-like, values in [0,1] (can be discrete / non-continuous).
        alpha, beta: Beta distribution parameters (>0).
        n_samples: number of samples to draw.
        replace: sample with replacement or not.
        eps: clip x into [eps, 1-eps] for numerical stability.
        rng: optional np.random.Generator.
        return_probs: if True, also return the sampling probabilities p_i.

    Returns:
        samples: resampled values from x
        (optional) probs: sampling probability for each element in x
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if not (alpha > 0 and beta > 0):
        raise ValueError("alpha and beta must be > 0.")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    if not np.all((0.0 <= x) & (x <= 1.0)):
        raise ValueError("All x must be in [0,1].")

    if rng is None:
        rng = np.random.default_rng()

    # clip to avoid log(0)
    xc = np.clip(x, eps, 1.0 - eps)

    # log weight: (alpha-1)log(x) + (beta-1)log(1-x)
    logw = (alpha - 1.0) * np.log(xc) + (beta - 1.0) * np.log1p(-xc)

    # stable softmax to get probabilities
    logw = logw - np.max(logw)
    w = np.exp(logw)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        raise RuntimeError("Weight normalization failed. Try different alpha/beta or larger eps.")
    p = w / w_sum

    idx = rng.choice(len(x), size=n_samples, replace=replace, p=p)
    samples = x[idx]

    if return_probs:
        return samples, p
    return samples


def _as_1d_array(x: Any) -> np.ndarray:
    if isinstance(x, (list, tuple)) and len(x) > 0 and hasattr(x[0], "reshape"):
        return np.asarray(x[0]).reshape(-1)
    return np.asarray(x).reshape(-1)


def load_x_cal(directory: str) -> Dict[Any, Dict[str, Any]]:
    x_cal = {}
    for filename in os.listdir(directory):
        if not filename.endswith(".pkl"):
            continue
        file_path = os.path.join(directory, filename)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        for idx, entry in data.items():
            key = idx
            detoxify_ft_value = entry.get("detoxify_ft")
            detoxify_human_value = entry.get("detoxify_human", {}).get("toxicity")

            if detoxify_ft_value is None or detoxify_human_value is None:
                continue

            x_cal[key] = {
                "pred": entry.get("pred"),
                "detoxify_ft": detoxify_ft_value,
                "detoxify_human": detoxify_human_value,
            }
    return x_cal


def load_y_cal(conformal_path: str) -> Dict[Any, Any]:
    with open(conformal_path, "rb") as f:
        y_cal = pickle.load(f)
    return y_cal


def _adversarial_alpha(t: int, T: int) -> float:
    """Piecewise linear adversarial ramp: 0.5 -> 1 -> 3 over [1, T]."""
    t_norm = (t - 1) / (T - 1) if T > 1 else 0.0
    t_third = (T / 3.0 - 1) / (T - 1) if T > 1 else 1.0 / 3.0
    if t_norm <= t_third:
        alpha = 0.5 + 0.5 * (t_norm / t_third) if t_third > 0 else 0.5
    else:
        alpha = 1.0 + 2.0 * ((t_norm - t_third) / (1.0 - t_third)) if t_third < 1.0 else 1.0
    return float(alpha)


def get_beta_alpha_at_t(
    t: int,
    T: int,
    setting: str,
    jump_schedule: Optional[List[Tuple[int, int, float]]] = None,
) -> float:
    """
    Get beta distribution alpha parameter at time t.

    Args:
        t: current time step (1-indexed)
        T: total time steps
        setting: "uniform", "adversarial", or "adversarial_jump"
        jump_schedule: for "adversarial_jump" only — list of (start_t, end_t, alpha)
            triples defining short jump windows (non-overlapping, all after t=1000).
            Outside these windows the smooth adversarial ramp is used.

    Returns:
        alpha parameter for beta distribution
    """
    if setting == "uniform":
        return 1.0
    elif setting == "adversarial":
        return _adversarial_alpha(t, T)
    elif setting == "adversarial_jump":
        if jump_schedule is None:
            raise ValueError("jump_schedule must be provided for 'adversarial_jump' setting.")
        # If t falls inside any jump window, return that window's alpha
        for start_t, end_t, jump_alpha in jump_schedule:
            if start_t <= t <= end_t:
                return float(jump_alpha)
        # Otherwise follow the smooth adversarial ramp
        return _adversarial_alpha(t, T)
    else:
        raise ValueError(f"Unknown setting: {setting}")


def sample_calibration_and_test_sets(
    all_keys: List[Any],
    tox_scores: Dict[Any, float],
    t: int,
    T: int,
    cal_size: int,
    test_size: int,
    beta_setting: str,
    beta_param: float,  # beta parameter (b) for beta distribution
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    jump_schedule: Optional[List[Tuple[int, int, float]]] = None,
) -> Tuple[List[Any], List[Any]]:
    """
    Sample calibration and test sets using beta-weighted resampling.

    Args:
        all_keys: all available keys
        tox_scores: toxicity scores for each key
        t: current time step
        T: total time steps
        cal_size: size of calibration set
        test_size: size of test set
        beta_setting: "uniform", "adversarial", or "adversarial_jump"
        beta_param: beta parameter (b) for beta distribution (alpha varies with t)
        seed: random seed
        rng: optional random generator
        jump_schedule: for "adversarial_jump" — list of (end_t, alpha) pairs

    Returns:
        (calibration_keys, test_keys)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # Get toxicity scores as array (normalized to [0,1])
    tox_array = np.array([tox_scores.get(k, 0.0) for k in all_keys], dtype=float)

    # Normalize to [0,1] if needed (assuming toxicity scores are already in [0,1])
    if tox_array.max() > 1.0 or tox_array.min() < 0.0:
        # Normalize to [0,1]
        tox_min = tox_array.min()
        tox_max = tox_array.max()
        if tox_max > tox_min:
            tox_array = (tox_array - tox_min) / (tox_max - tox_min)
        else:
            tox_array = np.zeros_like(tox_array)

    # Get alpha parameter for this time step
    alpha_t = get_beta_alpha_at_t(t, T, beta_setting, jump_schedule=jump_schedule)
    
    # Compute sampling probabilities based on beta distribution weights
    # Weight is proportional to Beta(alpha_t, beta_param) PDF evaluated at toxicity score
    tox_clipped = np.clip(tox_array, 1e-6, 1.0 - 1e-6)
    logw = (alpha_t - 1.0) * np.log(tox_clipped) + (beta_param - 1.0) * np.log1p(-tox_clipped)
    logw = logw - np.max(logw)  # Numerical stability
    w = np.exp(logw)
    p = w / w.sum()
    
    # Sample indices for calibration and test sets based on beta-weighted probabilities
    cal_indices = rng.choice(len(all_keys), size=cal_size, replace=True, p=p)
    test_indices = rng.choice(len(all_keys), size=test_size, replace=True, p=p)
    
    cal_keys = [all_keys[i] for i in cal_indices]
    test_keys = [all_keys[i] for i in test_indices]
    
    return cal_keys, test_keys


def prompt_toxicity_summary(x_cal: Dict[Any, Dict[str, Any]], mode: str = "max") -> Dict[Any, float]:
    out = {}
    for k, v in x_cal.items():
        human = _as_1d_array(v["detoxify_human"])
        if human.size == 0:
            out[k] = float("nan")
            continue
        if mode == "max":
            out[k] = float(np.max(human))
        elif mode == "mean":
            out[k] = float(np.mean(human))
        else:
            raise ValueError("mode must be 'max' or 'mean'")
    return out


def make_pool(keys: List[Any], pool_frac: float, seed: int) -> List[Any]:
    rng = np.random.default_rng(seed)
    n = len(keys)
    m = max(1, int(np.floor(pool_frac * n)))
    idx = rng.choice(n, size=m, replace=False)
    return [keys[i] for i in idx]


def sample_prompt_sequence(
    pool_keys: List[Any],
    tox_score: Dict[Any, float],
    T: int,
    seed: int,
    walk_sigma: float,
    temp: float,
) -> Tuple[List[Any], np.ndarray]:
    """
    s_t random walk in [0,1].
    Use s_t to bias toward high-tox or low-tox prompts:
      weight ∝ exp( z_i * (2*s_t-1) / temp )
    where z_i is standardized toxicity within the pool.
    """
    rng = np.random.default_rng(seed)

    tox = np.array([tox_score[k] for k in pool_keys], dtype=float)
    tox = np.nan_to_num(tox, nan=np.nanmedian(tox))
    z = (tox - tox.mean()) / (tox.std() + 1e-12)

    s = 0.5
    s_hist = np.zeros(T, dtype=float)
    seq = []

    for t in range(T):
        s = float(np.clip(s + rng.normal(0.0, walk_sigma), 0.0, 1.0))
        s_hist[t] = s

        direction = (2.0 * s - 1.0)  # <0 prefers low tox, >0 prefers high tox
        logits = (direction * z) / max(temp, 1e-6)
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()

        idx = int(rng.choice(len(pool_keys), p=p))
        seq.append(pool_keys[idx])

    return seq, s_hist


def loss_from_lambda_for_prompt(
    x_cal: Dict[Any, Dict[str, Any]],
    prompt_key: Any,
    lam: float,
    empty_loss: float = 0.0,
    mode: str = "fixed_budget",
    budget: int = 4,
) -> float:
    """
    Compute loss for a prompt given lambda threshold.
    Loss is the maximum human toxicity among responses with detoxify_ft <= lam.
    """
    if prompt_key not in x_cal:
        # raise ValueError(f"Prompt key {prompt_key} not found in x_cal")
        return float(empty_loss)
    
    v = x_cal[prompt_key]
    ft = _as_1d_array(v["detoxify_ft"])
    # ft = _as_1d_array(v["detoxify_human"])
    human = _as_1d_array(v["detoxify_human"])
    
    if ft.size == 0 or human.size == 0:
        # raise ValueError(f"Prompt key {prompt_key} has no detoxify_ft or detoxify_human")
        return float(empty_loss)
    
    # Find indices where detoxify_ft <= lam
    selected = [idx for idx in range(min(ft.size, human.size)) 
                if 0 <= idx < ft.size and ft[idx] <= lam]
    
    if len(selected) == 0:
        # print(f"Prompt key {prompt_key} has no detoxify_ft <= lam")
        return float(empty_loss)
    
    if mode == "max":
    # Get human toxicity values for selected indices
        vals = [human[idx] for idx in selected if 0 <= idx < human.size]
        if len(vals) == 0:
            return float(empty_loss)
        return float(np.max(vals))
    elif mode == "first":
        return float(human[selected[0]])
    elif mode == "fixed_budget":
        selected = [idx for idx in range(min(ft.size, human.size)) 
                if 0 <= idx < budget and ft[idx] <= lam]
        if len(selected) == 0:
            return float(empty_loss)
        vals = [human[idx] for idx in selected if 0 <= idx < human.size]
        if len(vals) == 0:
            return float(empty_loss)
        return float(human[selected[0]])

def empirical_cvar(losses: np.ndarray, beta: float) -> float:
    if losses.size == 0:
        return np.nan
    k = max(1, int(np.ceil((1.0 - beta) * losses.size)))
    idx = np.argpartition(losses, -k)[-k:]
    return float(np.mean(losses[idx]))


@dataclass
class OnlineCVaRControl:
    """RU Conformal Inference (Algorithm: RU Conformal Inference).

    Outer level: coordinate-descent threshold (CDT) update on lambda.
    Inner level: AdaGrad--FTRL update on the Rockafellar--Uryasev (RU)
    auxiliary variable c, whose argmin is the regularized leader over the
    full loss history.
    """
    beta: float
    alpha: float
    gamma: float
    lambda0: float          # initial lambda_1
    T: int
    lam_min: float = 0.002
    lam_max: float = 0.991

    lambda_hist: List[float] = field(default_factory=list, init=False)
    c_hist: List[float] = field(default_factory=list, init=False)
    cvar_est_hist: List[float] = field(default_factory=list, init=False)
    loss_hist: List[float] = field(default_factory=list, init=False)

    def project(self, lam: float) -> float:
        return float(min(self.lam_max, max(self.lam_min, lam)))

    @staticmethod
    def _ftrl_argmin_c(
        sorted_losses: List[float],
        eta: float,
        inv_tail: float,
        t: int,
        iters: int = 80,
    ) -> float:
        """Exact minimizer over c in [0, 1] of the regularized leader

            (1 / (2 eta)) (c - 1/2)^2
                + sum_{s=1}^{t} [ c + inv_tail * (R_s - c)_+ ].

        The objective is convex with non-decreasing derivative
            f'(c) = (c - 1/2)/eta + t - inv_tail * #{s : R_s > c},
        so we bisect for its zero crossing. `sorted_losses` must hold the
        R_s(lambda_s), s=1..t, in ascending order.
        """
        def fprime(c: float) -> float:
            m = t - bisect.bisect_right(sorted_losses, c)  # #{s : R_s > c}
            return (c - 0.5) / eta + t - inv_tail * m

        lo, hi = 0.0, 1.0
        if fprime(lo) >= 0.0:
            return lo
        if fprime(hi) <= 0.0:
            return hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if fprime(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def run(self, Rt) -> Dict[str, np.ndarray]:
        if not (0.0 <= self.beta < 1.0):
            raise ValueError("beta must be in [0,1).")
        inv_tail = 1.0 / (1.0 - self.beta)

        # --- Initialization ---
        lam = float(self.lambda0)                                # lambda_1
        c = 0.5                                                   # c_1
        q = max(1.0, self.beta / (1.0 - self.beta)) ** 2         # q_0
        sorted_losses: List[float] = []                          # R_s(lambda_s), ascending

        for t in range(1, self.T + 1):
            # Nature move + loss extension: evaluate R_t at lambda_t clamped
            # to [lam_min, lam_max]. For lambda_t outside the range this equals
            # R_min,t (lambda_t < lam_min) / R_max,t (lambda_t > lam_max), since
            # the per-prompt loss is monotone non-decreasing in lambda.
            lam_eval = self.project(lam)
            L_t = float(Rt(lam_eval, t))

            # RU surrogate CVaR estimate at (c_t, lambda_t)
            cvar_hat = c + inv_tail * max(0.0, L_t - c)

            # Record the values actually used at step t.
            self.lambda_hist.append(lam)
            self.c_hist.append(c)
            self.cvar_est_hist.append(cvar_hat)
            self.loss_hist.append(L_t)

            # Outer-level CDT update -> lambda_{t+1}
            lam = lam - self.gamma * (cvar_hat - self.alpha)

            # Inner-level AdaGrad--FTRL update -> c_{t+1}
            indicator = 1.0 if (L_t > c) else 0.0
            g_t = 1.0 - inv_tail * indicator
            q = q + g_t * g_t
            eta = 1.0 / (2.0 * np.sqrt(q))

            bisect.insort(sorted_losses, L_t)
            c = self._ftrl_argmin_c(sorted_losses, eta, inv_tail, t)

        return {
            "lambda": np.array(self.lambda_hist, dtype=float),
            "c": np.array(self.c_hist, dtype=float),
            "CVaR_est": np.array(self.cvar_est_hist, dtype=float),
            "loss": np.array(self.loss_hist, dtype=float),
        }


def distortion_risk_control_online(
    x_cal: Dict[Any, Dict[str, Any]],
    y_cal: Dict[Any, List[Tuple]],
    alpha: float,
    beta: float,
    lam_min: float = 0.002,
    lam_max: float = 0.991,
    n_candidates: int = 1000,
) -> Optional[float]:
    """
    Select the largest static lambda such that the estimated CVaR of per-prompt
    max-human-toxicity (over conformal-set responses with detoxify_ft <= lambda)
    is controlled at level alpha, with beta-CVaR and a normal confidence correction.

    Mirrors distortion_risk_control() from sampling_detoxify_0.7.py but uses
    the rank-normalized detoxify_ft scores already stored in x_cal, and takes
    y_cal[key] = list of (response_idx, text) tuples (no outer 'set' wrapper).
    """
    lambda_candidates = np.linspace(lam_min, lam_max, n_candidates)
    risks = []

    for lam in tqdm(lambda_candidates, desc="Static lambda selection"):
        r_lambdas = []
        for key, val in x_cal.items():
            if key not in y_cal:
                continue
            ft = _as_1d_array(val["detoxify_ft"])
            human = _as_1d_array(val["detoxify_human"])
            C_all = y_cal[key]
            C_lambda = [idx for idx, _ in C_all if 0 <= idx < ft.size and ft[idx] <= lam]
            if len(C_lambda) == 0:
                continue
            r_lambdas.append(max(human[idx] for idx in C_lambda if 0 <= idx < human.size))

        if len(r_lambdas) == 0:
            risks.append(np.inf)
            continue

        r_arr = np.array(r_lambdas)
        var_r = np.percentile(r_arr, beta * 100)
        tail = [r for r in r_arr if r > var_r]
        empirical_cvar_val = float(np.mean(tail)) if tail else float(var_r)
        max_values = np.maximum(r_arr, var_r)
        sigma = 1.0 / (1.0 - beta) * np.std(max_values)
        risks.append(empirical_cvar_val + 1.645 * sigma / np.sqrt(len(r_arr)))

    risks = np.array(risks)
    valid = lambda_candidates[risks <= alpha]
    return float(np.max(valid)) if valid.size > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--conformal_path", type=str, required=True)
    parser.add_argument("--conformal_key", type=str, default=None)

    parser.add_argument("--T", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=100)

    # CVaR control parameters
    parser.add_argument("--beta", type=float, default=0.75, help="CVaR beta parameter (confidence level)")
    parser.add_argument("--alpha", type=float, default=0.35, help="Target CVaR alpha")
    parser.add_argument("--gamma", type=float, default=0.05, help="Learning rate for lambda update")
    parser.add_argument("--lambda0", type=float, default=0.5, help="Initial lambda (lambda_1)")

    parser.add_argument("--lam_min", type=float, default=0.002)
    parser.add_argument("--lam_max", type=float, default=0.991)
    parser.add_argument("--empty_loss", type=float, default=0.0)
    # Sampling parameters
    parser.add_argument("--tox_mode", type=str, default="mean", choices=["max", "mean"], 
                       help="How to summarize toxicity scores per prompt")
    parser.add_argument("--cal_size", type=int, default=500, help="Calibration set size")
    parser.add_argument("--test_size", type=int, default=1000, help="Test set size")
    parser.add_argument("--beta_setting", type=str, default="uniform",
                       choices=["uniform", "adversarial", "adversarial_jump"],
                       help="Beta distribution setting: uniform, adversarial (smooth ramp), or adversarial_jump (abrupt jumps)")
    parser.add_argument("--beta_param", type=float, default=1.0,
                       help="Beta distribution beta parameter (b)")
    # adversarial_jump parameters
    parser.add_argument("--n_jumps", type=int, default=5,
                       help="Number of jump windows for adversarial_jump setting")
    parser.add_argument("--jump_alphas", type=str, default="0.3,5.0,0.3,5.0,0.3",
                       help="Comma-separated alpha values per jump window (length must equal n_jumps)")
    parser.add_argument("--jump_max_duration", type=int, default=100,
                       help="Maximum duration (steps) of each jump window (default: 30)")
    parser.add_argument("--jump_seed", type=int, default=None,
                       help="Seed for sampling random jump positions (defaults to --seed)")

    parser.add_argument("--out_dir", type=str, default="./outputs_online_detoxify")
    parser.add_argument("--window", type=int, default=200,
                       help="Window size for the moving/rolling CVaR across steps (mirrors invest sliding window)")
    args = parser.parse_args()

    # Build jump_schedule for adversarial_jump setting
    jump_schedule = None
    if args.beta_setting == "adversarial_jump":
        jump_alphas = [float(v) for v in args.jump_alphas.split(",")]
        if len(jump_alphas) != args.n_jumps:
            raise ValueError(
                f"--jump_alphas has {len(jump_alphas)} values but --n_jumps={args.n_jumps}. "
                "They must match."
            )
        d = args.jump_max_duration
        jump_start = 1001  # jumps only allowed after t=1000
        # To guarantee non-overlap, space start times at least (d+1) apart.
        # Map n_jumps starts into [jump_start, T - d] with min gap (d+1) using the
        # order-statistics trick: sample offsets in reduced range then shift by i*(d+1).
        min_start, max_start = jump_start, args.T - d
        effective_range = max_start - min_start - args.n_jumps * (d + 1)
        if effective_range < 0:
            raise ValueError(
                f"Not enough room for {args.n_jumps} non-overlapping jumps of max duration {d} "
                f"after t=1000 with T={args.T}. Reduce n_jumps or jump_max_duration."
            )
        jump_rng = np.random.default_rng(args.jump_seed if args.jump_seed is not None else args.seed)
        raw = sorted(jump_rng.choice(effective_range + 1, size=args.n_jumps, replace=False).tolist())
        start_times = [min_start + raw[i] + i * (d + 1) for i in range(args.n_jumps)]
        # Each jump has a random duration in [1, d]
        jump_schedule = []
        for i, st in enumerate(start_times):
            duration = int(jump_rng.integers(1, d + 1))
            jump_schedule.append((st, st + duration - 1, jump_alphas[i]))
        print(f"\nAdversarial jump schedule ({args.n_jumps} windows, max duration {d}, all after t=1000):")
        for st, et, alpha in jump_schedule:
            print(f"  t in [{st}, {et}] (len={et - st + 1}) -> alpha={alpha} (base ramp otherwise)")
        print()

    # Load data
    x_cal = load_x_cal(args.directory)
    
    # Get all available keys and compute toxicity scores
    all_keys = sorted(list(x_cal.keys()))
    if len(all_keys) == 0:
        raise ValueError("No data loaded from directory")
    
    tox_score = prompt_toxicity_summary(x_cal, mode=args.tox_mode)
    
    # Remove keys with NaN toxicity scores
    valid_keys = [k for k in all_keys if not np.isnan(tox_score.get(k, np.nan))]
    if len(valid_keys) == 0:
        raise ValueError("No keys with valid toxicity scores")
    
    print(f"Loaded {len(valid_keys)} prompts with valid toxicity scores")
    
    # ============================================================
    # Transform machine scores to normalized ranks (empirical CDF)
    # ============================================================
    # Step 1: Collect all machine scores with their (prompt_key, response_idx) location
    all_machine_scores = []
    score_locations = []  # (prompt_key, response_idx)
    
    for k in valid_keys:
        v = x_cal[k]
        machine_scores = _as_1d_array(v["detoxify_ft"])
        for idx, score in enumerate(machine_scores):
            all_machine_scores.append(score)
            score_locations.append((k, idx))
    
    all_machine_scores = np.array(all_machine_scores)
    total_scores = len(all_machine_scores)
    
    print(f"\n{'='*60}")
    print("Machine Score Rank Transformation")
    print(f"{'='*60}")
    print(f"Total number of responses: {total_scores}")
    print(f"Original machine scores - Min: {all_machine_scores.min():.6f}, Max: {all_machine_scores.max():.6f}, Mean: {all_machine_scores.mean():.6f}")
    
    # Step 2: Compute ranks (1 = smallest, N = largest) and normalize to [0, 1]
    # Using 'average' method for ties (same as scipy.stats.rankdata default)
    ranks = rankdata(all_machine_scores, method='average')
    normalized_ranks = ranks / total_scores  # Now in (0, 1]
    
    print(f"Normalized ranks - Min: {normalized_ranks.min():.6f}, Max: {normalized_ranks.max():.6f}, Mean: {normalized_ranks.mean():.6f}")
    
    # Step 3: Replace detoxify_ft in x_cal with normalized ranks
    # First, group normalized ranks back by prompt
    prompt_normalized_scores = {k: [] for k in valid_keys}
    for i, (k, idx) in enumerate(score_locations):
        prompt_normalized_scores[k].append(normalized_ranks[i])
    
    # Update x_cal with normalized ranks
    for k in valid_keys:
        original_len = len(_as_1d_array(x_cal[k]["detoxify_ft"]))
        x_cal[k]["detoxify_ft_original"] = x_cal[k]["detoxify_ft"]  # Keep original
        x_cal[k]["detoxify_ft"] = np.array(prompt_normalized_scores[k])
        assert len(x_cal[k]["detoxify_ft"]) == original_len, f"Length mismatch for {k}"
    
    print(f"Replaced detoxify_ft with normalized ranks for all prompts")
    print(f"{'='*60}")
    
    # ============================================================
    # Compute per-prompt quantiles for human and machine scores
    # ============================================================
    percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Store per-prompt quantiles: {percentile: [list of values per prompt]}
    human_quantiles = {p: [] for p in percentiles}
    machine_quantiles = {p: [] for p in percentiles}
    
    for k in valid_keys:
        v = x_cal[k]
        human_scores = _as_1d_array(v["detoxify_human"])
        machine_scores = _as_1d_array(v["detoxify_ft"])
        
        if len(human_scores) > 0:
            for p in percentiles:
                human_quantiles[p].append(np.quantile(human_scores, p))
        
        if len(machine_scores) > 0:
            for p in percentiles:
                machine_quantiles[p].append(np.quantile(machine_scores, p))
    
    print(f"\n{'='*70}")
    print("Per-Prompt Quantile Analysis (Mean over prompts)")
    print(f"{'='*70}")
    print(f"Number of prompts: {len(valid_keys)}")
    print(f"\n{'Percentile':<12} {'Human (mean)':<15} {'Human (std)':<15} {'Machine (mean)':<15} {'Machine (std)':<15}")
    print("-" * 70)
    for p in percentiles:
        h_arr = np.array(human_quantiles[p])
        m_arr = np.array(machine_quantiles[p])
        print(f"{p:<12.1f} {h_arr.mean():<15.6f} {h_arr.std():<15.6f} {m_arr.mean():<15.6f} {m_arr.std():<15.6f}")
    print(f"{'='*70}")
    
    # Use 25% quantile (0.25 -> interpolate between 0.2 and 0.3) for fixed lambda
    # Or directly compute 0.25 quantile
    per_prompt_q25 = []
    for k in valid_keys:
        v = x_cal[k]
        machine_scores = _as_1d_array(v["detoxify_ft"])
        if len(machine_scores) > 0:
            per_prompt_q25.append(np.quantile(machine_scores, 0.25))
    
    per_prompt_q25 = np.array(per_prompt_q25)
    lambda_fixed_25q = np.mean(per_prompt_q25)
    
    print(f"\nFixed Lambda (avg of per-prompt 25% quantile of machine scores): {lambda_fixed_25q:.6f}")
    
    # Compute loss for each prompt using this fixed lambda
    losses_fixed_lambda = []
    prompts_with_loss = []
    for k in valid_keys:
        try:
            loss = loss_from_lambda_for_prompt(
                x_cal=x_cal,
                prompt_key=k,
                lam=lambda_fixed_25q,
                empty_loss=args.empty_loss,
                mode="fixed_budget",
            )
            losses_fixed_lambda.append(loss)
            prompts_with_loss.append(k)
        except Exception as e:
            # Skip prompts that have no responses <= lambda
            pass
    
    losses_fixed_lambda = np.array(losses_fixed_lambda)
    
    print(f"\nNumber of prompts with valid loss (at least one response <= lambda): {len(losses_fixed_lambda)}")
    print(f"Loss statistics:")
    print(f"  Mean loss: {losses_fixed_lambda.mean():.6f}")
    print(f"  Std loss: {losses_fixed_lambda.std():.6f}")
    print(f"  Min loss: {losses_fixed_lambda.min():.6f}")
    print(f"  Max loss: {losses_fixed_lambda.max():.6f}")
    
    # Compute CVaR at different beta levels
    for beta_level in [0.5, 0.75, 0.9, 0.95]:
        cvar_val = empirical_cvar(losses_fixed_lambda, beta_level)
        print(f"  CVaR(beta={beta_level:.2f}): {cvar_val:.6f}")
    
    print(f"{'='*60}\n")
    
    # Initialize random generator
    rng = np.random.default_rng(args.seed)
    
    # Store calibration and test sets for each time step
    cal_sets = []
    test_sets = []
    test_prompt_keys = []  # The prompt key used at each time step (from test set)
    cal_tox_scores_list = []  # Store toxicity scores of calibration sets
    alpha_values = []  # Store alpha values at each time step
    
    # Pre-sample all calibration and test sets
    print(f"Pre-sampling calibration and test sets for T={args.T} steps...")
    for t in range(1, args.T + 1):
        alpha_t = get_beta_alpha_at_t(t, args.T, args.beta_setting, jump_schedule=jump_schedule)
        alpha_values.append(alpha_t)

        cal_keys, test_keys = sample_calibration_and_test_sets(
            all_keys=valid_keys,
            tox_scores=tox_score,
            t=t,
            T=args.T,
            cal_size=args.cal_size,
            test_size=args.test_size,
            beta_setting=args.beta_setting,
            beta_param=args.beta_param,
            seed=args.seed + t,  # Different seed for each time step
            rng=rng,
            jump_schedule=jump_schedule,
        )
        cal_sets.append(cal_keys)
        test_sets.append(test_keys)
        
        # Store toxicity scores of calibration set
        cal_tox = [tox_score.get(k, 0.0) for k in cal_keys]
        cal_tox_scores_list.append(cal_tox)
        
        # Sample one prompt from test set for this time step
        test_prompt_keys.append(rng.choice(test_keys))
    
    print("Sampling complete.")

    # ============================================================
    # Static Lambda Baseline: select lambda once on pooled cal set
    # using distortion risk control (mirrors sampling_detoxify_0.7.py)
    # ============================================================
    y_cal_raw = load_y_cal(args.conformal_path)
    # Flatten: y_cal_sets[key] = [(idx, text), ...]
    y_cal_sets = {k: v["set"] for k, v in y_cal_raw.items()}

    pooled_cal_keys = list(set(k for keys in cal_sets for k in keys))
    pooled_cal_keys = [k for k in pooled_cal_keys if k in x_cal and k in y_cal_sets]
    print(f"Pooled calibration keys for static lambda: {len(pooled_cal_keys)} unique prompts")

    x_cal_static = {k: x_cal[k] for k in pooled_cal_keys}
    y_cal_static = {k: y_cal_sets[k] for k in pooled_cal_keys}

    lambda_static = distortion_risk_control_online(
        x_cal=x_cal_static,
        y_cal=y_cal_static,
        alpha=args.alpha,
        beta=args.beta,
        lam_min=args.lam_min,
        lam_max=args.lam_max,
    )
    if lambda_static is None:
        lambda_static = args.lam_min
        print(f"No valid static lambda found; using lam_min={args.lam_min}")
    else:
        print(f"Static lambda (distortion risk control): {lambda_static:.6f}")

    # Define loss function: at time t, use test set prompt and calibration set
    def Rt(lam: float, t: int) -> float:
        # Get the test prompt for this time step
        prompt_key = test_prompt_keys[t - 1]
        return loss_from_lambda_for_prompt(
            x_cal=x_cal,
            prompt_key=prompt_key,
            lam=lam,
            empty_loss=args.empty_loss,
        )

    algo = OnlineCVaRControl(
        beta=args.beta,
        alpha=args.alpha,
        gamma=args.gamma,
        lambda0=args.lambda0,
        T=args.T,
        lam_min=args.lam_min,
        lam_max=args.lam_max,
    )
    out = algo.run(Rt)

    losses = out["loss"]
    realized = np.array([empirical_cvar(losses[:t], args.beta) for t in range(1, len(losses) + 1)], dtype=float)

    # Baseline: apply static lambda to the same test sequence
    losses_static = np.array([Rt(lambda_static, t) for t in range(1, args.T + 1)], dtype=float)
    realized_static = np.array([empirical_cvar(losses_static[:t], args.beta) for t in range(1, args.T + 1)], dtype=float)

    # ------------------------------------------------------------
    # Moving (sliding-window) CVaR across steps, mirroring
    # online_cvar_invest.py. At step t (0-indexed), use the last
    # `window` losses: losses[t-window+1 : t+1]. NaN until the
    # window has filled. Also track a rolling mean of the raw loss.
    # ------------------------------------------------------------
    window = max(1, int(args.window))
    n_steps = len(losses)
    rolling_cvar = np.full(n_steps, np.nan)
    rolling_cvar_static = np.full(n_steps, np.nan)
    rolling_mean_loss = np.full(n_steps, np.nan)
    for t in range(window - 1, n_steps):
        lo = t - window + 1
        rolling_cvar[t] = empirical_cvar(losses[lo:t + 1], args.beta)
        rolling_cvar_static[t] = empirical_cvar(losses_static[lo:t + 1], args.beta)
        rolling_mean_loss[t] = float(np.mean(losses[lo:t + 1]))


    # Compute beta alpha values over time
    beta_alpha_hist = np.array([get_beta_alpha_at_t(t, args.T, args.beta_setting, jump_schedule=jump_schedule)
                                for t in range(1, args.T + 1)])
    cs = out["c"]
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame({
        "t": np.arange(1, args.T + 1),
        "prompt_key": test_prompt_keys,
        "lambda": out["lambda"],
        "c": out["c"],
        "CVaR_est": out["CVaR_est"],
        "loss": losses,
        "CVaR_realized": realized,
        "beta_alpha": beta_alpha_hist,
        "lambda_static": lambda_static,
        "loss_static": losses_static,
        "CVaR_realized_static": realized_static,
        "CVaR_rolling": rolling_cvar,
        "CVaR_rolling_static": rolling_cvar_static,
        "rolling_mean_loss": rolling_mean_loss,
    })
    tag = f"T{args.T}_beta{args.beta}_alpha{args.alpha}_setting{args.beta_setting}_cal{args.cal_size}_test{args.test_size}"
    csv_path = os.path.join(args.out_dir, f"online_{tag}.csv")
    df.to_csv(csv_path, index=False)

    subtitle = f"β={args.beta}, α={args.alpha}, γ={args.gamma}, setting={args.beta_setting}, T={args.T}, cal={args.cal_size}"

    # ---- time-series plots (notebook style; first PLOT_BURN_IN steps dropped) ----
    t_axis = np.arange(1, args.T + 1)
    t_vals = t_axis  # reused by the calibration plots below
    bi = PLOT_BURN_IN if args.T > PLOT_BURN_IN else 0
    sl = slice(bi, None)
    t_plot = t_axis[sl]

    # Realized CVaR
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t_plot, realized[sl], color=COLORS["portfolio"], linewidth=LINE_WIDTH,
            label=fr"Realized CVaR (β={args.beta})")
    ax.plot(t_plot, realized_static[sl], color=COLORS["baseline"], linewidth=LINE_WIDTH,
            linestyle="-.", label=fr"Static λ={lambda_static:.3f}")
    ax.axhline(args.alpha, color=COLORS["target"], linestyle="--", linewidth=2.4,
               label=fr"Target α={args.alpha}")
    _style_axes(ax, "Step", "CVaR of Loss")
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(args.out_dir, f"cvar_{tag}.png"))

    # Rolling (sliding-window) CVaR
    fig, ax = plt.subplots(figsize=FIGSIZE)
    valid_roll = ~np.isnan(rolling_cvar)
    ax.plot(t_axis[valid_roll], rolling_cvar[valid_roll], color=COLORS["portfolio"],
            linewidth=LINE_WIDTH, label=fr"{window}-step Rolling CVaR (β={args.beta})")
    valid_roll_s = ~np.isnan(rolling_cvar_static)
    ax.plot(t_axis[valid_roll_s], rolling_cvar_static[valid_roll_s], color=COLORS["baseline"],
            linewidth=LINE_WIDTH, linestyle="-.",
            label=fr"{window}-step Rolling CVaR — Static λ={lambda_static:.3f}")
    ax.axhline(args.alpha, color=COLORS["target"], linestyle="--", linewidth=2.4,
               label=fr"Target α={args.alpha}")
    _style_axes(ax, "Step", "CVaR of Loss")
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(args.out_dir, f"cvar_rolling_{tag}.png"))

    # c_t
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t_plot, cs[sl], color=COLORS["secondary"], linewidth=LINE_WIDTH, label=r"$c_t$")
    ax.axhline(args.alpha, color=COLORS["target"], linestyle="--", linewidth=2.4,
               label=fr"Target α={args.alpha}")
    _style_axes(ax, "Step", "c")
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(args.out_dir, f"cts_{tag}.png"))

    # Lambda
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t_plot, out["lambda"][sl], color=COLORS["primary"], linewidth=LINE_WIDTH, label=r"$\lambda_t$")
    _style_axes(ax, "Step", "Lambda")
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(args.out_dir, f"lambda_{tag}.png"))

    # Beta-distribution a-parameter over time
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t_plot, beta_alpha_hist[sl], color=COLORS["portfolio"], linewidth=LINE_WIDTH,
            label=fr"$a_t$ ({args.beta_setting})")
    _style_axes(ax, "Step", r"$a_t$")
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(args.out_dir, f"beta_alpha_{tag}.png"))

    # Plot calibration set toxicity scores distribution at different alpha values
    # Sample a few representative time points to show
    if args.beta_setting == "adversarial":
        # Sample at different alpha values: low (t=1), medium (t=T/3), high (t=T)
        sample_indices = [0, max(0, args.T // 3 - 1), args.T - 1]
        sample_labels = ["t=1 (α≈0.5)", f"t=T/3 (α≈1.0)", f"t=T (α≈3.0)"]
    else:
        # For uniform, just sample a few evenly spaced points
        sample_indices = [0, args.T // 2, args.T - 1]
        sample_labels = [f"t={i+1} (α=1.0)" for i in sample_indices]
    
    # Create subplot for toxicity distributions
    fig, axes = plt.subplots(1, len(sample_indices), figsize=(6 * len(sample_indices), 5))
    if len(sample_indices) == 1:
        axes = [axes]
    
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    for idx, (sample_idx, label) in enumerate(zip(sample_indices, sample_labels)):
        if sample_idx >= len(cal_tox_scores_list):
            continue
        
        cal_tox = np.array(cal_tox_scores_list[sample_idx])
        alpha_val = alpha_values[sample_idx]
        
        axes[idx].hist(cal_tox, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black', linewidth=1.2)
        axes[idx].axvline(cal_tox.mean(), color='#E63946', linestyle='--', linewidth=2, 
                         label=f'Mean={cal_tox.mean():.3f}')
        axes[idx].axvline(np.median(cal_tox), color='#F18F01', linestyle='--', linewidth=2, 
                         label=f'Median={np.median(cal_tox):.3f}')
        axes[idx].set_xlabel("Human Toxicity Score", fontsize=11, fontweight='bold')
        axes[idx].set_ylabel("Frequency", fontsize=11, fontweight='bold')
        axes[idx].set_title(f"{label}\nα={alpha_val:.3f}, n={len(cal_tox)}", 
                           fontsize=12, fontweight='bold')
        axes[idx].legend(loc='best', fontsize=9, framealpha=0.9)
        axes[idx].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.suptitle(
        f"Calibration Set Toxicity Distribution at Different α Values ({args.beta_setting} setting)\n{subtitle}",
        fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"cal_tox_dist_{tag}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot mean/median toxicity of calibration sets vs alpha
    cal_tox_means = [np.mean(cal_tox) for cal_tox in cal_tox_scores_list]
    cal_tox_medians = [np.median(cal_tox) for cal_tox in cal_tox_scores_list]
    cal_tox_stds = [np.std(cal_tox) for cal_tox in cal_tox_scores_list]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Top plot: Mean/Median toxicity vs alpha
    ax1.plot(alpha_values, cal_tox_means, 
             label="Mean Toxicity", 
             linewidth=2.5,
             color='#E63946',
             marker='o',
             markersize=5,
             markevery=max(1, len(alpha_values) // 20),
             alpha=0.9)
    ax1.plot(alpha_values, cal_tox_medians, 
             label="Median Toxicity", 
             linewidth=2.5,
             color='#F18F01',
             marker='s',
             markersize=5,
             markevery=max(1, len(alpha_values) // 20),
             alpha=0.9)
    ax1.fill_between(alpha_values,
                     np.array(cal_tox_means) - np.array(cal_tox_stds),
                     np.array(cal_tox_means) + np.array(cal_tox_stds),
                     alpha=0.2, color='#E63946', label='±1 Std')
    ax1.set_xlabel("Beta α Parameter", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Toxicity Score", fontsize=12, fontweight='bold')
    ax1.set_title(f"Calibration Set Toxicity Statistics vs α ({args.beta_setting} setting)\n{subtitle}",
                  fontsize=12, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Bottom plot: Mean/Median toxicity vs time
    ax2.plot(t_vals, cal_tox_means, 
             label="Mean Toxicity", 
             linewidth=2.5,
             color='#E63946',
             marker='o',
             markersize=5,
             markevery=max(1, len(t_vals) // 20),
             alpha=0.9)
    ax2.plot(t_vals, cal_tox_medians, 
             label="Median Toxicity", 
             linewidth=2.5,
             color='#F18F01',
             marker='s',
             markersize=5,
             markevery=max(1, len(t_vals) // 20),
             alpha=0.9)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t_vals, alpha_values,
                  label="α_t",
                  linewidth=2,
                  color='#2E86AB',
                  linestyle=':',
                  alpha=0.7)
    ax2.set_xlabel("Time Step (t)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Toxicity Score", fontsize=12, fontweight='bold', color='#E63946')
    ax2_twin.set_ylabel("α_t", fontsize=12, fontweight='bold', color='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#E63946')
    ax2_twin.tick_params(axis='y', labelcolor='#2E86AB')
    ax2.set_title("Calibration Set Toxicity Statistics Over Time", 
                  fontsize=14, fontweight='bold', pad=15)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"cal_tox_vs_alpha_{tag}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Done.")
    print(f"Beta setting: {args.beta_setting}")
    print(f"Calibration set size: {args.cal_size}")
    print(f"Test set size: {args.test_size}")
    print("saved:", csv_path)
    print("final realized CVaR:", realized[-1])


if __name__ == "__main__":
    main()
