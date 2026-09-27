from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn

import untappd
import untappd_utils

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError on Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass
class FriendStats:
    name: str
    n_checkins: int
    raw_mean: float
    raw_std: float
    bayes_mean: float
    bayes_ci_low: float
    bayes_ci_high: float
    mean_abv: float
    abv_adj_mean: float
    abv_adj_ci_low: float
    abv_adj_ci_high: float
    solo_p_value: float


def compute_friend_statistics(
    checkins: Sequence[untappd.Checkin],
    min_checkins: int = 5,
) -> tuple[dict[str, float], list[FriendStats], list[untappd.Checkin]]:
    """
    Computes rigorous empirical Bayes shrinkage and ABV-adjusted residuals for friends.
    """
    valid_checkins = [
        c
        for c in checkins
        if c.rating is not None and c.beer.abv is not None and c.rating > 0
    ]
    ratings = np.array([c.rating for c in valid_checkins], dtype=float)
    abvs = np.array([c.beer.abv for c in valid_checkins], dtype=float)

    # 1. Global Baseline & OLS for ABV Control
    mu_0 = float(np.mean(ratings))
    var_0 = float(np.var(ratings, ddof=1))
    slope_abv, intercept_abv, r_val, p_val, std_err = stats.linregress(abvs, ratings)

    solo_ratings = [
        c.rating
        for c in valid_checkins
        if not c.tagged_friends or not c.tagged_friends.strip()
    ]
    solo_mean = float(np.mean(solo_ratings)) if solo_ratings else mu_0
    solo_var = float(np.var(solo_ratings, ddof=1)) if solo_ratings else var_0

    # Residual relative to ABV expectation: R_i - E[R | ABV_i]
    abv_expected = intercept_abv + slope_abv * abvs
    abv_residuals = ratings - abv_expected
    var_resid_0 = float(np.var(abv_residuals, ddof=1))

    # Map each checkin to its residuals
    resid_lookup = {id(c): r for c, r in zip(valid_checkins, abv_residuals)}

    # Group checkins by individual friend
    by_friend_ratings: dict[str, list[float]] = defaultdict(list)
    by_friend_abvs: dict[str, list[float]] = defaultdict(list)
    by_friend_resids: dict[str, list[float]] = defaultdict(list)

    for c in valid_checkins:
        if c.tagged_friends:
            friend_names = [f.strip() for f in c.tagged_friends.split(",") if f.strip()]
            for name in friend_names:
                by_friend_ratings[name].append(c.rating)
                by_friend_abvs[name].append(c.beer.abv)
                by_friend_resids[name].append(resid_lookup[id(c)])

    # Compute empirical Bayes posterior estimates
    stats_list: list[FriendStats] = []
    prec_prior = 1.0 / var_0
    prec_resid_prior = 1.0 / var_resid_0

    for name, f_ratings in by_friend_ratings.items():
        n = len(f_ratings)
        if n < min_checkins:
            continue

        y_bar = float(np.mean(f_ratings))
        s2 = float(np.var(f_ratings, ddof=1)) if n > 1 else var_0
        f_abvs = by_friend_abvs[name]
        mean_abv = float(np.mean(f_abvs))

        # Conjugate normal-normal posterior for raw rating
        prec_data = n / s2 if s2 > 1e-6 else n / var_0
        post_var = 1.0 / (prec_prior + prec_data)
        post_mean = post_var * (prec_prior * mu_0 + prec_data * y_bar)
        post_se = np.sqrt(post_var)

        # ABV-adjusted residual empirical Bayes
        f_resids = by_friend_resids[name]
        res_bar = float(np.mean(f_resids))
        res_s2 = float(np.var(f_resids, ddof=1)) if n > 1 else var_resid_0
        prec_res_data = n / res_s2 if res_s2 > 1e-6 else n / var_resid_0
        post_res_var = 1.0 / (prec_resid_prior + prec_res_data)
        post_res_mean = post_res_var * (
            prec_resid_prior * 0.0 + prec_res_data * res_bar
        )
        post_res_se = np.sqrt(post_res_var)

        # Welch's t-test vs Solo checkins
        t_stat, p_val_solo = stats.ttest_ind(f_ratings, solo_ratings, equal_var=False)

        stats_list.append(
            FriendStats(
                name=name,
                n_checkins=n,
                raw_mean=y_bar,
                raw_std=float(np.std(f_ratings, ddof=1)) if n > 1 else 0.0,
                bayes_mean=post_mean,
                bayes_ci_low=post_mean - 1.96 * post_se,
                bayes_ci_high=post_mean + 1.96 * post_se,
                mean_abv=mean_abv,
                abv_adj_mean=post_res_mean,
                abv_adj_ci_low=post_res_mean - 1.96 * post_res_se,
                abv_adj_ci_high=post_res_mean + 1.96 * post_res_se,
                solo_p_value=float(p_val_solo),
            )
        )

    # Sort by Bayesian posterior mean rating descending
    stats_list.sort(key=lambda s: s.bayes_mean, reverse=True)

    meta = {
        "mu_0": mu_0,
        "var_0": var_0,
        "solo_mean": solo_mean,
        "solo_n": float(len(solo_ratings)),
        "slope_abv": slope_abv,
        "intercept_abv": intercept_abv,
        "r2_abv": float(r_val**2),
    }
    return meta, stats_list, valid_checkins


@untappd_utils.show_or_save_to_out_file
def plot_friend_analytics(
    checkins: Sequence[untappd.Checkin],
    min_checkins: int = 10,
    out_file: Optional[Path] = None,
):
    """
    Renders a comprehensive four-panel figure visualizing social check-in dynamics.
    Panel A: Empirical Bayes credible intervals for raw ratings vs population prior.
    Panel B: Rating vs Mean Session ABV (disentangling alcohol strength from friend preference).
    Panel C: ABV-adjusted rating residuals (true taste delta when controlling for ABV).
    Panel D: Sample size N vs Uncertainty interval width (demonstrating shrinkage power).
    """
    meta, stats_list, valid_checkins = compute_friend_statistics(
        checkins, min_checkins=min_checkins
    )

    if not stats_list:
        print(f"No friends found with at least {min_checkins} checkins.")
        return

    seaborn.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(
        2, 2, figsize=(18, 14), gridspec_kw={"hspace": 0.28, "wspace": 0.24}
    )
    ((ax1, ax2), (ax3, ax4)) = axes

    names = [s.name for s in stats_list]
    y_pos = np.arange(len(names))

    # --- PANEL A: Empirical Bayes Shrinkage of Mean Ratings ---
    bayes_means = np.array([s.bayes_mean for s in stats_list])
    raw_means = np.array([s.raw_mean for s in stats_list])
    ci_lows = np.array([s.bayes_ci_low for s in stats_list])
    ci_highs = np.array([s.bayes_ci_high for s in stats_list])
    x_err = np.array([bayes_means - ci_lows, ci_highs - bayes_means])

    ax1.errorbar(
        bayes_means,
        y_pos,
        xerr=x_err,
        fmt="o",
        color="#2b5c8f",
        ecolor="#4a90e2",
        elinewidth=2.2,
        capsize=4,
        capthick=1.5,
        markersize=6.5,
        label="Empirical Bayes Posterior Mean (95% CI)",
        zorder=3,
    )
    ax1.scatter(
        raw_means,
        y_pos,
        marker="x",
        color="#d9534f",
        s=45,
        linewidths=1.8,
        label="Raw Sample Mean",
        zorder=4,
    )
    ax1.axvline(
        meta["solo_mean"],
        color="#6c757d",
        linestyle="--",
        linewidth=1.8,
        label=f"Solo Drinking Mean ({meta['solo_mean']:.3f})",
        zorder=2,
    )
    ax1.axvline(
        meta["mu_0"],
        color="#28a745",
        linestyle=":",
        linewidth=1.8,
        label=f"Dataset Overall Mean ({meta['mu_0']:.3f})",
        zorder=2,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(
        [f"{s.name} (N={s.n_checkins})" for s in stats_list],
        fontsize=9.5,
        fontweight="medium",
    )
    ax1.invert_yaxis()
    ax1.set_xlabel("Rating (0 to 5)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"A. Friend Rating Estimation (Empirical Bayes Shrinkage, N >= {min_checkins})",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax1.legend(loc="lower right", framealpha=0.92, fontsize=9)

    # --- PANEL B: Mean ABV vs Mean Rating (The 'Heavy Drinking Friend' Effect) ---
    abvs = np.array([s.mean_abv for s in stats_list])
    counts = np.array([s.n_checkins for s in stats_list])
    # Normalize sizes for bubble scatter
    bubble_sizes = 40 + (np.log1p(counts) / np.log1p(max(counts))) * 360

    scatter = ax2.scatter(
        abvs,
        raw_means,
        s=bubble_sizes,
        c=raw_means,
        cmap="viridis",
        edgecolors="#333333",
        linewidths=1.2,
        alpha=0.88,
        zorder=3,
    )
    # Regression line for friends
    x_abv_grid = np.linspace(min(abvs) - 0.5, max(abvs) + 0.5, 100)
    ax2.plot(
        x_abv_grid,
        meta["intercept_abv"] + meta["slope_abv"] * x_abv_grid,
        color="#e67e22",
        linestyle="--",
        linewidth=2,
        label=f"Dataset ABV OLS Expectation (Slope: +{meta['slope_abv']:.3f}/ABV%)",
        zorder=2,
    )

    for s in stats_list:
        ax2.annotate(
            s.name,
            (s.mean_abv, s.raw_mean),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8.5,
            alpha=0.85,
        )

    ax2.set_xlabel("Mean Beer ABV (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Raw Mean Rating", fontsize=11, fontweight="bold")
    ax2.set_title(
        "B. Social Palate Bias: Does High ABV Explain Higher Ratings?",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    cbar = fig.colorbar(scatter, ax=ax2, pad=0.02)
    cbar.set_label("Raw Mean Rating", fontsize=9.5)
    ax2.legend(loc="upper left", framealpha=0.92, fontsize=9)

    # --- PANEL C: ABV-Adjusted Residuals (True Disentangled Taste Premium) ---
    res_stats = sorted(stats_list, key=lambda s: s.abv_adj_mean, reverse=True)
    res_names = [s.name for s in res_stats]
    res_y_pos = np.arange(len(res_names))
    res_means = np.array([s.abv_adj_mean for s in res_stats])
    res_ci_low = np.array([s.abv_adj_ci_low for s in res_stats])
    res_ci_high = np.array([s.abv_adj_ci_high for s in res_stats])
    res_x_err = np.array([res_means - res_ci_low, res_ci_high - res_means])

    bar_colors = ["#27ae60" if rm >= 0 else "#e74c3c" for rm in res_means]
    ax3.errorbar(
        res_means,
        res_y_pos,
        xerr=res_x_err,
        fmt="o",
        color="#2c3e50",
        ecolor="#7f8c8d",
        elinewidth=2.0,
        capsize=4,
        capthick=1.2,
        markersize=6,
        zorder=3,
    )
    ax3.scatter(
        res_means,
        res_y_pos,
        c=bar_colors,
        s=60,
        edgecolors="#2c3e50",
        zorder=4,
    )
    ax3.axvline(0.0, color="#34495e", linestyle="-", linewidth=1.5, zorder=2)
    ax3.set_yticks(res_y_pos)
    ax3.set_yticklabels(
        [f"{s.name} ({s.abv_adj_mean:+.3f})" for s in res_stats],
        fontsize=9.5,
        fontweight="medium",
    )
    ax3.invert_yaxis()
    ax3.set_xlabel(
        "Rating Residual Relative to ABV Expectation [R - E(R|ABV)]",
        fontsize=11,
        fontweight="bold",
    )
    ax3.set_title(
        "C. ABV-Controlled Quality Effect (Net Friend Palate Residual)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )

    # --- PANEL D: Statistical Uncertainty & Credible Interval Width vs Sample Size ---
    ci_widths = [s.bayes_ci_high - s.bayes_ci_low for s in stats_list]
    ax4.scatter(
        counts,
        ci_widths,
        s=80,
        color="#8e44ad",
        edgecolors="#4a154b",
        linewidths=1.2,
        alpha=0.85,
        zorder=3,
    )
    ax4.set_xscale("log")
    ax4.set_xlabel("Check-in Sample Size N (Log Scale)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("95% Credible Interval Width", fontsize=11, fontweight="bold")
    ax4.set_title(
        "D. Epistemic Uncertainty vs. Observation Frequency",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )

    for s, w in zip(stats_list, ci_widths):
        if s.n_checkins > 70 or w > 0.25:
            ax4.annotate(
                s.name,
                (s.n_checkins, w),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=8.5,
                alpha=0.85,
            )

    plt.suptitle(
        f"Untappd Social Dynamics: Friend Rating Estimation, ABV Disentanglement & Uncertainty (N >= {min_checkins})",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )


def print_summary_table(stats_list: list[FriendStats], meta: dict[str, float]):
    """Prints a structured summary table to stdout."""
    print("=" * 105)
    print(f"{'FRIEND RATING & SOCIAL DYNAMICS SUMMARY':^105}")
    print("=" * 105)
    print(
        f"Population Baseline: N={int(meta['solo_n'])} solo checkins (Mean: {meta['solo_mean']:.3f})"
    )
    print(
        f"ABV OLS Model: Rating = {meta['intercept_abv']:.3f} + {meta['slope_abv']:.4f} * ABV (R2 = {meta['r2_abv']:.3f})"
    )
    print("-" * 105)
    print(
        f"{'Friend Name':22s} {'N':>5s} {'Raw':>6s} {'Bayes':>6s} {'95% CI':>16s} {'Mean ABV':>9s} {'ABV Resid':>10s} {'p (vs Solo)':>12s}"
    )
    print("-" * 105)
    for s in stats_list:
        ci_str = f"[{s.bayes_ci_low:.3f}, {s.bayes_ci_high:.3f}]"
        sig = (
            "***"
            if s.solo_p_value < 0.001
            else (
                "**"
                if s.solo_p_value < 0.01
                else ("*" if s.solo_p_value < 0.05 else "ns")
            )
        )
        print(
            f"{s.name:22s} {s.n_checkins:5d} {s.raw_mean:6.3f} {s.bayes_mean:6.3f} {ci_str:>16s} {s.mean_abv:8.2f}% {s.abv_adj_mean:+9.3f}   {s.solo_p_value:8.4f} ({sig})"
        )
    print("=" * 105)


if __name__ == "__main__":
    checkins = untappd.load_latest_checkins()
    meta, stats_list, _ = compute_friend_statistics(checkins, min_checkins=10)
    print_summary_table(stats_list, meta)

    out_path = Path(__file__).parent / "out" / "friend_ratings.png"
    plot_friend_analytics(checkins, min_checkins=10, out_file=out_path)
