import argparse
import bisect
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ----------------------------------------------------------------------------
# Plotting style ported from the plots notebook: large bold fonts, dark grid,
# fixed palette, YYYY date axis, no titles. PLOT_BURN_IN drops the first rows
# from PLOTS ONLY (the saved CSV keeps every post-burn-in row).
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


def _style_axes(ax, xlabel: str, ylabel: str, date_axis: bool = False, year_step: int = 2) -> None:
    ax.figure.patch.set_facecolor(FIG_FACE)
    ax.set_facecolor(AX_FACE)
    ax.set_title("")
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    if date_axis:
        ax.xaxis.set_major_locator(mdates.YearLocator(year_step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.35)


def _finalize(fig, ax, path: str, date_axis: bool = False) -> None:
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_SIZE)
    if date_axis:
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def simple_return_from_close(close: np.ndarray, horizon: int) -> np.ndarray:
    """
    r_t^(h) = close[t] / close[t-h] - 1, aligned to start at index h
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    close = np.asarray(close, dtype=float)
    return np.log(close[horizon:] / close[:-horizon])


def yield_to_simple_return(yield_annual_pct: np.ndarray, periods_per_year: int) -> np.ndarray:
    """
    Convert annualized yield in percent to per-period simple return approx.
    r ≈ y / 100 / periods_per_year
    """
    y = np.asarray(yield_annual_pct, dtype=float)
    return y / 100.0 / float(periods_per_year)


def empirical_cvar(losses: np.ndarray, beta: float) -> float:
    """
    Empirical CVaR_beta of losses (higher = worse), i.e., mean of worst (1-beta) tail.
    """
    if losses.size == 0:
        return np.nan
    k = max(1, int(np.ceil((1.0 - beta) * losses.size)))
    idx = np.argpartition(losses, -k)[-k:]
    return float(np.mean(losses[idx]))

def print_baseline_cvar_stats(
    dates: pd.Series,
    risky_ret: np.ndarray,
    rf_ret: np.ndarray,
    betas: List[float],
    horizon: int,
    year: int = 2009,
    lam_list=(0.0, 0.25, 0.5, 0.75, 1.0),
):
    """
    Print baseline CVaR stats for the given return series.
    Loss = -(lambda * risky + (1-lambda) * rf)
    """
    # optionally filter by year
    if year is not None:
        mask = pd.to_datetime(dates).dt.year == year
        risky_ret = np.asarray(risky_ret)[mask.values]
        rf_ret = np.asarray(rf_ret)[mask.values]
        dates = dates[mask.values]

    n = len(risky_ret)
    if n == 0:
        print("[WARN] No data available for baseline CVaR printing.")
        return

    print("\n" + "=" * 80)
    if year is None:
        print(f"Baseline CVaR on FULL data | horizon={horizon} | N={n}")
    else:
        print(f"Baseline CVaR on YEAR={year} | horizon={horizon} | N={n}")

    print(f"Risky ret: mean={risky_ret.mean():.6f}, std={risky_ret.std():.6f}")
    print(f"RF    ret: mean={rf_ret.mean():.6f}, std={rf_ret.std():.6f}")

    # Print CVaR for risky returns
    risky_loss = -risky_ret
    print(f"\n--- Risky Asset CVaR ---")
    risky_q95 = float(np.quantile(risky_loss, 0.95))
    risky_max = float(np.max(risky_loss))
    risky_mean = float(np.mean(risky_loss))
    print(f"Loss mean={risky_mean:.6f} | 95% quantile={risky_q95:.6f} | max={risky_max:.6f}")
    risky_betas = [0.75, 0.8, 0.85, 0.9]
    for beta in risky_betas:
        cvar = empirical_cvar(risky_loss, beta)
        print(f"  CVaR(beta={beta:.3f}) = {cvar:.6f}")

    for lam in lam_list:
        rp = lam * risky_ret + (1.0 - lam) * rf_ret
        loss = -rp
        q95 = float(np.quantile(loss, 0.95))
        mx = float(np.max(loss))
        mu = float(np.mean(loss))

        print(f"\n--- baseline lambda={lam:.2f} ---")
        print(f"Loss mean={mu:.6f} | 95% quantile={q95:.6f} | max={mx:.6f}")
        for beta in betas:
            cvar = empirical_cvar(loss, beta)
            print(f"  CVaR(beta={beta:.3f}) = {cvar:.6f}")

    print("\n[Hint] alpha should be around CVaR of your intended baseline (e.g., lambda=0.5)")
    print("=" * 80 + "\n")

@dataclass
class OnlineCVaRControlInvest:
    """RU Conformal Inference for an investment portfolio.

    lambda_t is the risky weight; the loss is
        L_t(lambda) = -(lambda * r_risky_t + (1-lambda) * r_rf_t),
    extended outside [lam_min, lam_max] by its inf (below) / sup (above)
    over the range. Outer level: CDT update on lambda. Inner level:
    AdaGrad--FTRL update on the Rockafellar--Uryasev variable c in [0,1].

    Losses are expected to be normalized to ~[0,1] (see run_one_setting),
    since c_1 = 1/2 and c is constrained to [0,1].
    """
    beta: float
    alpha: float
    gamma: float
    lambda0: float          # initial lambda_1
    lam_min: float
    lam_max: float

    r_risky: np.ndarray  # length T
    r_rf: np.ndarray     # length T

    # logs
    lam_hist: List[float] = field(default_factory=list, init=False)
    lam_used_hist: List[float] = field(default_factory=list, init=False)  # lambda used to compute returns
    c_hist: List[float] = field(default_factory=list, init=False)
    cvar_est_hist: List[float] = field(default_factory=list, init=False)
    loss_hist: List[float] = field(default_factory=list, init=False)
    wealth_hist: List[float] = field(default_factory=list, init=False)

    def project_lam(self, lam: float) -> float:
        return float(min(self.lam_max, max(self.lam_min, lam)))

    @staticmethod
    def _ftrl_argmin_c(
        sorted_losses: List[float],
        eta: float,
        inv_tail: float,
        t: int,
        iters: int = 80,
    ) -> float:
        """Exact minimizer over c in [0,1] of the regularized leader

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

    def run(self) -> Dict[str, np.ndarray]:
        if not (0.0 <= self.beta < 1.0):
            raise ValueError("beta must be in [0,1).")
        inv_tail = 1.0 / (1.0 - self.beta)

        T = len(self.r_risky)
        if len(self.r_rf) != T:
            raise ValueError("r_risky and r_rf must have same length.")

        # --- Initialization (RU Conformal Inference) ---
        lam = float(self.lambda0)                              # lambda_1
        c = 0.5                                                # c_1
        q = max(1.0, self.beta / (1.0 - self.beta)) ** 2       # q_0
        sorted_losses: List[float] = []                        # R_s(lambda_s), ascending

        wealth = 1.0

        for t in range(1, T + 1):
            rr = float(self.r_risky[t - 1])
            rf = float(self.r_rf[t - 1])

            # lambda used to realize this period's return (pre-update)
            lam_used = lam

            # Loss extension over [lam_min, lam_max]: L_t(lam) is linear in lam,
            # so its inf/sup over the range are attained at the endpoints.
            L_at_min = -(self.lam_min * rr + (1.0 - self.lam_min) * rf)
            L_at_max = -(self.lam_max * rr + (1.0 - self.lam_max) * rf)
            R_min_t, R_max_t = min(L_at_min, L_at_max), max(L_at_min, L_at_max)
            if lam < self.lam_min:
                L_t = R_min_t
            elif lam > self.lam_max:
                L_t = R_max_t
            else:
                L_t = -(lam * rr + (1.0 - lam) * rf)

            # RU surrogate CVaR estimate at (c_t, lambda_t)
            cvar_hat = c + inv_tail * max(0.0, L_t - c)

            # realized portfolio return + wealth use the (raw) lambda actually held
            rp = lam_used * rr + (1.0 - lam_used) * rf
            wealth *= np.exp(rp)

            # Outer-level CDT update -> lambda_{t+1}
            lam = lam - self.gamma * (cvar_hat - self.alpha)

            # logs (lambda_used = lambda held this step; lambda = next-period lambda)
            self.lam_hist.append(lam)
            self.lam_used_hist.append(lam_used)
            self.c_hist.append(c)
            self.cvar_est_hist.append(cvar_hat)
            self.loss_hist.append(L_t)
            self.wealth_hist.append(wealth)

            # Inner-level AdaGrad--FTRL update -> c_{t+1}
            indicator = 1.0 if (L_t > c) else 0.0
            g_t = 1.0 - inv_tail * indicator
            q = q + g_t * g_t
            eta = 1.0 / (2.0 * np.sqrt(q))
            bisect.insort(sorted_losses, L_t)
            c = self._ftrl_argmin_c(sorted_losses, eta, inv_tail, t)

        return {
            "lambda": np.array(self.lam_hist, dtype=float),  # lambda for next period
            "lambda_used": np.array(self.lam_used_hist, dtype=float),  # lambda used for return calculation
            "c": np.array(self.c_hist, dtype=float),
            "CVaR_est": np.array(self.cvar_est_hist, dtype=float),
            "loss": np.array(self.loss_hist, dtype=float),
            "wealth": np.array(self.wealth_hist, dtype=float),
        }


def run_one_setting(
    df: pd.DataFrame,
    beta: float,
    alpha: float,
    gamma: float,
    lambda0: float,
    lam_min: float,
    lam_max: float,
    out_dir: str,
    tag: str,
    burn_in_days: int = 50,
    normalize_losses: bool = True,
    baseline_lambda: float = None,
) -> None:
    r_risky = df["r_risky"].to_numpy(dtype=float)
    r_rf = df["r_rf"].to_numpy(dtype=float)

    # Normalize losses to [0, 1] using 1% and 99% quantiles of pooled rf/rr losses.
    # RU Conformal Inference expects losses ~[0,1] (c_1=1/2, c in [0,1]), so keep
    # this on unless you know the raw losses already live in [0,1].
    if normalize_losses:
        l_rf = -r_rf
        l_rr = -r_risky
        all_losses = np.concatenate([l_rf, l_rr])
        q1_loss = float(np.quantile(all_losses, 0.01))
        q99_loss = float(np.quantile(all_losses, 0.99))
        scale = q99_loss - q1_loss
        if scale < 1e-12:
            raise ValueError("Loss scale is near-zero; cannot normalize.")
        # Normalizing: l_norm = (l - q1) / scale  =>  r_norm = -l_norm = (r + q1) / scale
        r_risky_algo = (r_risky + q1_loss) / scale
        r_rf_algo = (r_rf + q1_loss) / scale
        alpha_algo = (alpha - q1_loss) / scale
        print(f"[normalize] q1={q1_loss:.6f}, q99={q99_loss:.6f}, scale={scale:.6f}")
        print(f"[normalize] alpha: {alpha:.6f} -> {alpha_algo:.6f}")
    else:
        print("[WARN] normalize_losses is OFF; RU assumes losses in [0,1] (c_1=1/2, c in [0,1]).")
        r_risky_algo, r_rf_algo = r_risky, r_rf
        alpha_algo = alpha
        q1_loss, scale = 0.0, 1.0

    algo = OnlineCVaRControlInvest(
        beta=beta,
        alpha=alpha_algo,
        gamma=gamma,
        lambda0=baseline_lambda,
        lam_min=lam_min,
        lam_max=lam_max,
        r_risky=r_risky_algo,
        r_rf=r_rf_algo,
    )
    out = algo.run()

    # Denormalize loss, c, CVaR_est back to original scale: x_orig = x_norm * scale + q1
    out["loss"] = out["loss"] * scale + q1_loss
    out["c"] = out["c"] * scale + q1_loss
    out["CVaR_est"] = out["CVaR_est"] * scale + q1_loss

    losses = out["loss"]
    # Compute realized CVaR excluding burn-in period
    losses_post_burnin = losses[burn_in_days:]
    realized_cvar_post_burnin = np.array(
        [empirical_cvar(losses_post_burnin[:t], beta) for t in range(1, len(losses_post_burnin) + 1)],
        dtype=float
    )
    # Final empirical CVaR across all post-burn-in periods
    final_cvar = empirical_cvar(losses_post_burnin, beta)

    # Calculate portfolio returns and cumulative returns (excluding burn-in for plots)
    # Use lambda_used (the lambda that was actually used to compute returns) instead of lambda (next period's lambda)
    portfolio_ret = out["lambda_used"] * df["r_risky"].values + (1.0 - out["lambda_used"]) * df["r_rf"].values
    
    # Slice data to exclude burn-in period for plotting
    portfolio_ret_post = portfolio_ret[burn_in_days:]
    r_risky_post = df["r_risky"].values[burn_in_days:]
    r_rf_post = df["r_rf"].values[burn_in_days:]
    dates_post = df["date"].values[burn_in_days:]
    
    cumulative_portfolio = np.exp(np.cumsum(portfolio_ret_post))
    cumulative_risky = np.exp(np.cumsum(r_risky_post))
    cumulative_rf = np.exp(np.cumsum(r_rf_post))
    
    # Calculate losses separately for risky and risk-free assets (post burn-in)
    loss_risky = -r_risky_post
    loss_rf = -r_rf_post

    # Sliding window CVaR: at t >= 179, CVaR of losses[t-179 : t+1] (180-day window)
    window = 500
    n_post = len(losses_post_burnin)
    sliding_cvar = np.full(n_post, np.nan)
    for t in range(window - 1, n_post):
        sliding_cvar[t] = empirical_cvar(losses_post_burnin[t - window + 1 : t + 1], beta)

    # Baseline (fixed lambda) series for comparison
    bl_cvar_realized = np.full(n_post, np.nan)
    bl_sliding_cvar = np.full(n_post, np.nan)
    bl_cumulative = np.full(n_post, np.nan)
    bl_loss = np.full(n_post, np.nan)
    if baseline_lambda is not None:
        bl_loss_arr = -(baseline_lambda * r_risky_post + (1.0 - baseline_lambda) * r_rf_post)
        bl_loss = bl_loss_arr
        bl_cvar_realized = np.array(
            [empirical_cvar(bl_loss_arr[:t], beta) for t in range(1, n_post + 1)],
            dtype=float,
        )
        for t in range(window - 1, n_post):
            bl_sliding_cvar[t] = empirical_cvar(bl_loss_arr[t - window + 1 : t + 1], beta)
        bl_cumulative = np.exp(np.cumsum(-bl_loss_arr))

    res = pd.DataFrame({
        "date": dates_post,
        "r_risky": r_risky_post,
        "r_rf": r_rf_post,
        "r_portfolio": portfolio_ret_post,
        "lambda": out["lambda"][burn_in_days:],  # lambda for next period (what will be used)
        "lambda_used": out["lambda_used"][burn_in_days:],  # lambda actually used for this period's return
        "c": out["c"][burn_in_days:],
        "CVaR_est": out["CVaR_est"][burn_in_days:],
        "loss": losses_post_burnin,
        "loss_risky": loss_risky,
        "loss_rf": loss_rf,
        "CVaR_realized": realized_cvar_post_burnin,
        "CVaR_sliding_180": sliding_cvar,
        "CVaR_baseline": bl_cvar_realized,
        "CVaR_sliding_baseline": bl_sliding_cvar,
        "cumulative_baseline": bl_cumulative,
        "loss_baseline": bl_loss,
        "wealth": out["wealth"][burn_in_days:],
        "cumulative_portfolio": cumulative_portfolio,
        "cumulative_risky": cumulative_risky,
        "cumulative_rf": cumulative_rf,
    })

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"result_{tag}.csv")
    res.to_csv(csv_path, index=False)

    # ---- plots (notebook style; first PLOT_BURN_IN rows dropped from PLOTS only) ----
    res_plot = res.iloc[PLOT_BURN_IN:].reset_index(drop=True) if len(res) > PLOT_BURN_IN else res
    dates = pd.to_datetime(res_plot["date"].values)
    bl_final_cvar = empirical_cvar(bl_loss, beta) if baseline_lambda is not None else None

    # Realized CVaR
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dates, res_plot["CVaR_realized"].values, color=COLORS["portfolio"], linewidth=LINE_WIDTH,
            label=fr"Realized CVaR (β={beta})")
    if baseline_lambda is not None:
        ax.plot(dates, res_plot["CVaR_baseline"].values, color=COLORS["baseline"], linewidth=LINE_WIDTH,
                linestyle="-.", label="Baseline CVaR")
        ax.axhline(bl_final_cvar, color=COLORS["baseline"], linestyle=":", linewidth=3.0,
                   label=f"Baseline average CVaR={bl_final_cvar:.4f}")
    ax.axhline(alpha, color=COLORS["target"], linestyle="--", linewidth=2.4, label=fr"Target α={alpha}")
    ax.axhline(final_cvar, color=COLORS["full"], linestyle=":", linewidth=3.0,
               label=f"Full-period CVaR={final_cvar:.4f}")
    _style_axes(ax, "Date", "CVaR of Loss", date_axis=True)
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(out_dir, f"cvar_{tag}.png"), date_axis=True)

    # Rolling (sliding-window) CVaR
    fig, ax = plt.subplots(figsize=FIGSIZE)
    valid_sw = ~np.isnan(res_plot["CVaR_sliding_180"].values)
    ax.plot(dates[valid_sw], res_plot["CVaR_sliding_180"].values[valid_sw], color=COLORS["portfolio"],
            linewidth=LINE_WIDTH, label=fr"{window}-day Rolling CVaR (β={beta})")
    if baseline_lambda is not None:
        valid_bl = ~np.isnan(res_plot["CVaR_sliding_baseline"].values)
        if valid_bl.any():
            ax.plot(dates[valid_bl], res_plot["CVaR_sliding_baseline"].values[valid_bl],
                    color=COLORS["baseline"], linewidth=LINE_WIDTH, linestyle="-.",
                    label="Baseline Rolling CVaR")
        ax.axhline(bl_final_cvar, color=COLORS["baseline"], linestyle=":", linewidth=3.0,
                   label=f"Baseline average CVaR={bl_final_cvar:.4f}")
    ax.axhline(alpha, color=COLORS["target"], linestyle="--", linewidth=2.4, label=fr"Target α={alpha}")
    ax.axhline(final_cvar, color=COLORS["full"], linestyle=":", linewidth=3.0,
               label=f"Full-period CVaR={final_cvar:.4f}")
    _style_axes(ax, "Date", "CVaR of Loss", date_axis=True)
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(out_dir, f"cvar_rolling180_{tag}.png"), date_axis=True)

    # CVaR estimate
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dates, res_plot["CVaR_est"].values, color=COLORS["primary"], linewidth=LINE_WIDTH,
            label="Estimated CVaR")
    ax.axhline(alpha, color=COLORS["target"], linestyle="--", linewidth=2.4, label=fr"Target α={alpha}")
    _style_axes(ax, "Date", "CVaR", date_axis=True)
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(out_dir, f"cvar_est_{tag}.png"), date_axis=True)

    # c_t
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dates, res_plot["c"].values, color=COLORS["secondary"], linewidth=LINE_WIDTH, label=r"$c_t$")
    _style_axes(ax, "Date", "c", date_axis=True)
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(out_dir, f"ct_{tag}.png"), date_axis=True)

    # Lambda (risky weight actually used)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dates, res_plot["lambda_used"].values, color=COLORS["primary"], linewidth=LINE_WIDTH,
            label=r"$\lambda_t$")
    if baseline_lambda is not None:
        ax.axhline(baseline_lambda, color=COLORS["baseline"], linestyle="--", linewidth=2.2,
                   label=fr"Fixed λ={baseline_lambda:.2f}")
    _style_axes(ax, "Date", "Lambda", date_axis=True)
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(out_dir, f"lambda_{tag}.png"), date_axis=True)

    # Cumulative return curves
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dates, res_plot["cumulative_portfolio"].values, color=COLORS["portfolio"], linewidth=LINE_WIDTH,
            label="Portfolio")
    ax.plot(dates, res_plot["cumulative_risky"].values, color=COLORS["risky"], linewidth=LINE_WIDTH,
            linestyle="-.", label="Risky asset")
    ax.plot(dates, res_plot["cumulative_rf"].values, color=COLORS["rf"], linewidth=LINE_WIDTH,
            linestyle=":", label="Risk-free asset")
    if baseline_lambda is not None:
        ax.plot(dates, res_plot["cumulative_baseline"].values, color=COLORS["baseline"], linewidth=LINE_WIDTH,
                linestyle="--", label=fr"Baseline (λ={baseline_lambda:.2f})")
    ax.set_yscale("log")
    _style_axes(ax, "Date", "Cumulative Return", date_axis=True)
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(out_dir, f"returns_{tag}.png"), date_axis=True)

    # Portfolio loss
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dates, res_plot["loss"].values, color=COLORS["portfolio"], linewidth=LINE_WIDTH,
            label="Portfolio loss")
    if baseline_lambda is not None:
        ax.plot(dates, res_plot["loss_baseline"].values, color=COLORS["baseline"], linewidth=LINE_WIDTH,
                linestyle="-.", label="Baseline loss")
    _style_axes(ax, "Date", "Loss", date_axis=True)
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(out_dir, f"loss_{tag}.png"), date_axis=True)

    # Asset losses
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dates, res_plot["loss"].values, color=COLORS["portfolio"], linewidth=LINE_WIDTH,
            label="Portfolio loss")
    ax.plot(dates, res_plot["loss_risky"].values, color=COLORS["risky"], linewidth=LINE_WIDTH,
            linestyle="-.", label="Risky asset loss")
    ax.plot(dates, res_plot["loss_rf"].values, color=COLORS["rf"], linewidth=LINE_WIDTH,
            linestyle=":", label="Risk-free loss")
    _style_axes(ax, "Date", "Loss", date_axis=True)
    ax.legend(loc="best")
    _finalize(fig, ax, os.path.join(out_dir, f"loss_assets_{tag}.png"), date_axis=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", type=str, required=True, help="CSV with columns: date,risky_close,rf_close OR date,risky_close,rf_yield")
    p.add_argument("--date_col", type=str, default="date")
    p.add_argument("--risky_col", type=str, default="risky_close")
    p.add_argument("--rf_col", type=str, default="rf_close")

    p.add_argument("--horizon", type=int, default=1, help="Return horizon in days (e.g. 2 means two-day P&L)")
    p.add_argument("--rf_is_yield", action="store_true", help="Treat rf_col as annualized yield in percent, not a price")
    p.add_argument("--periods_per_year", type=int, default=252, help="Used when rf_is_yield is set")

    p.add_argument("--beta", type=str, required=True, help="Comma-separated betas, e.g. 0.75,0.9")
    p.add_argument("--alpha", type=str, required=True, help="Comma-separated alphas, e.g. 0.02,0.05")

    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--lambda0", type=float, default=0.5)

    p.add_argument("--lam_min", type=float, default=0.0)
    p.add_argument("--lam_max", type=float, default=1.0)

    p.add_argument("--out_dir", type=str, default="outputs_online_cvar")
    p.add_argument("--burn_in_days", type=int, default=0,
              help="Burn-in period length (days). During burn-in, lambda is fixed at lambda0 and empirical CVaR is estimated.")
    p.add_argument("--no_normalize_losses", action="store_true",
              help="Disable loss normalization to [0,1] via 1%%/99%% quantiles.")
    p.add_argument("--tune_days", type=int, default=0,
              help="Number of leading days to skip (tuning period). Both online method and baseline start from day tune_days+1.")
    p.add_argument("--baseline_lambda", type=float, default=None,
              help="Fixed lambda for the baseline portfolio. When set, adds baseline CVaR/return/loss curves to all plots.")
    
    # Period filtering
    p.add_argument("--year", type=int, default=None, help="Filter to specific year")
    p.add_argument("--quarter", type=int, default=None, choices=[1,2,3,4], help="Filter to specific quarter (1-4)")
    p.add_argument("--months", type=str, default=None, help="Comma-separated months (1-12), e.g. '1,2,3' for Q1")
    

    args = p.parse_args()

    df0 = pd.read_csv(args.data_csv)
    df0 = df0.rename(columns={
        args.date_col: "date",
        args.risky_col: "risky",
        args.rf_col: "rf",
    })

    df0["date"] = pd.to_datetime(df0["date"])
    df0 = df0.sort_values("date").reset_index(drop=True)

    risky_close = df0["risky"].to_numpy(dtype=float)

    if args.rf_is_yield:
        # rf is annualized yield (%); convert to daily simple return, then aggregate to horizon
        rf_daily = yield_to_simple_return(df0["rf"].to_numpy(dtype=float), args.periods_per_year)
        rf_daily_log = np.log1p(rf_daily)
        # make horizon return by compounding daily returns over horizon:
        # r^(h)_t = prod_{k=t-h+1..t} (1+r_k) - 1
        if args.horizon == 1:
            rf_ret = rf_daily_log[args.horizon:]  
            print(rf_ret)# align with risky returns after horizon handling below
        else:
            # build compounded horizon returns aligned to t>=h
            rf_h = np.array([rf_daily_log[t-args.horizon+1:t+1].sum()
                 for t in range(args.horizon, len(rf_daily_log))])
            rf_ret = rf_h
    else:
        rf_close = df0["rf"].to_numpy(dtype=float)
        rf_ret_full = simple_return_from_close(rf_close, args.horizon)
        rf_ret = rf_ret_full

    risky_ret = simple_return_from_close(risky_close, args.horizon)
    # align dates: returns start at index horizon
    dates = df0["date"].iloc[args.horizon:].reset_index(drop=True)

    # if rf_is_yield with horizon==1 we made rf_ret from rf_daily[1:], need align to dates
    if args.rf_is_yield and args.horizon == 1:
        # risky_ret length = N-horizon, rf_daily length = N, rf_daily[1:] length = N-1
        # but risky_ret starts at horizon=1 => length N-1, aligned
        pass

    if len(risky_ret) != len(rf_ret):
        n = min(len(risky_ret), len(rf_ret))
        risky_ret = risky_ret[:n]
        rf_ret = rf_ret[:n]
        dates = dates.iloc[:n]

    # Drop NaN days in rf_ret (and corresponding days in risky_ret)
    valid_mask = ~np.isnan(rf_ret)
    n_dropped = np.sum(~valid_mask)
    if n_dropped > 0:
        print(f"[INFO] Dropping {n_dropped} days with NaN in risk-free rate")
        risky_ret = risky_ret[valid_mask]
        rf_ret = rf_ret[valid_mask]
        dates = dates.iloc[valid_mask].reset_index(drop=True)

    # Filter by period (year, quarter, or months)
    if args.year is not None or args.quarter is not None or args.months is not None:
        period_mask = pd.Series([True] * len(dates), index=dates.index)
        
        if args.year is not None:
            period_mask = period_mask & (dates.dt.year == args.year)
        
        if args.quarter is not None:
            period_mask = period_mask & (dates.dt.quarter == args.quarter)
        
        if args.months is not None:
            months_list = [int(m.strip()) for m in args.months.split(",") if m.strip()]
            if not all(1 <= m <= 12 for m in months_list):
                raise ValueError("Months must be between 1 and 12")
            period_mask = period_mask & (dates.dt.month.isin(months_list))
        
        n_before = len(risky_ret)
        risky_ret = risky_ret[period_mask.values]
        rf_ret = rf_ret[period_mask.values]
        dates = dates.iloc[period_mask.values].reset_index(drop=True)
        n_after = len(risky_ret)
        
        period_desc = []
        if args.year:
            period_desc.append(f"year={args.year}")
        if args.quarter:
            period_desc.append(f"Q{args.quarter}")
        if args.months:
            period_desc.append(f"months={args.months}")
        print(f"[INFO] Filtered to {' '.join(period_desc)}: {n_before} -> {n_after} days")

    betas_for_print = parse_float_list(args.beta)  # use the same betas you will run
    # NOTE: dates already aligned to returns
    # Only pass year to baseline stats if not using period filtering (to avoid double filtering)
    baseline_year = args.year if (args.quarter is None and args.months is None) else None
    print_baseline_cvar_stats(
        dates=dates,
        risky_ret=risky_ret,
        rf_ret=rf_ret,
        betas=betas_for_print,
        horizon=args.horizon,
        year=baseline_year,
        lam_list=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    
    df = pd.DataFrame({
        "date": dates.values,
        "r_risky": risky_ret,
        "r_rf": rf_ret,
    })

    if args.tune_days > 0:
        if args.tune_days >= len(df):
            raise ValueError(
                f"tune_days={args.tune_days} >= total days={len(df)}. Reduce --tune_days."
            )
        print(f"[INFO] Skipping first {args.tune_days} tuning days. "
              f"Evaluation period: {len(df) - args.tune_days} days "
              f"({df['date'].iloc[args.tune_days].date()} to {df['date'].iloc[-1].date()})")
        df = df.iloc[args.tune_days:].reset_index(drop=True)

    betas = parse_float_list(args.beta)
    alphas = parse_float_list(args.alpha)

    for b in betas:
        for a in alphas:
            tag = f"beta{b}_alpha{a}_h{args.horizon}_g{args.gamma}"
            run_one_setting(
                df=df,
                beta=b,
                alpha=a,
                gamma=args.gamma,
                lambda0=args.baseline_lambda,
                lam_min=args.lam_min,
                lam_max=args.lam_max,
                out_dir=args.out_dir,
                tag=tag,
                burn_in_days=args.burn_in_days,
                normalize_losses=not args.no_normalize_losses,
                baseline_lambda=args.baseline_lambda,
            )

    print(f"Done. Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
