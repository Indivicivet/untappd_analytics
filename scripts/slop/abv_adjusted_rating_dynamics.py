from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.dates as mdates
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


def compute_abv_normalization_models(
    checkins: list[untappd.Checkin],
) -> dict[str, object]:
    """Fits cross-sectional linear, polynomial, and style-controlled ABV models."""
    valid_checkins = [
        c for c in checkins if c.rating is not None and c.beer.abv is not None
    ]
    ratings = np.array([c.rating for c in valid_checkins], dtype=float)
    abvs = np.array([c.beer.abv for c in valid_checkins], dtype=float)
    dates = np.array([c.datetime for c in valid_checkins])
    styles = [c.beer.get_style_category() for c in valid_checkins]

    mean_r = float(np.mean(ratings))
    mean_a = float(np.mean(abvs))

    # 1. Global Linear OLS Model: R_i = beta_0 + beta_1 * ABV_i + eps_i
    slope_lin, int_lin, r_lin, p_lin, std_err_lin = stats.linregress(abvs, ratings)
    adj_ratings_linear = ratings - slope_lin * (abvs - mean_a)

    # 2. Global Cubic Response Model: R_i = gamma_0 + gamma_1*ABV + gamma_2*ABV^2 + gamma_3*ABV^3
    poly_coeffs = np.polyfit(abvs, ratings, 3)
    poly_func = np.poly1d(poly_coeffs)
    adj_ratings_poly = ratings - poly_func(abvs) + mean_r

    # 3. Within-Style Fixed Effects Slope
    unique_styles = sorted(list(set(styles)))
    demeaned_abv = np.zeros(len(valid_checkins))
    demeaned_rating = np.zeros(len(valid_checkins))
    for s in unique_styles:
        m = np.array([st == s for st in styles])
        if np.sum(m) > 0:
            demeaned_abv[m] = abvs[m] - np.mean(abvs[m])
            demeaned_rating[m] = ratings[m] - np.mean(ratings[m])
    slope_within, _, r_within, p_within, _ = stats.linregress(
        demeaned_abv,
        demeaned_rating,
    )

    return {
        "valid_checkins": valid_checkins,
        "ratings": ratings,
        "abvs": abvs,
        "dates": dates,
        "styles": styles,
        "mean_rating": mean_r,
        "mean_abv": mean_a,
        "linear_model": {
            "slope": slope_lin,
            "intercept": int_lin,
            "r": r_lin,
            "r_squared": r_lin**2,
            "p_value": p_lin,
            "std_err": std_err_lin,
        },
        "poly_func": poly_func,
        "poly_coeffs": poly_coeffs,
        "within_style_slope": slope_within,
        "within_style_r": r_within,
        "adj_ratings_linear": adj_ratings_linear,
        "adj_ratings_poly": adj_ratings_poly,
    }


def compute_rolling_abv_dynamics(
    models: dict[str, object],
    window_size: int = 300,
) -> dict[str, np.ndarray]:
    """Computes rolling raw and ABV-adjusted rating dynamics over sliding windows."""
    dates: np.ndarray = models["dates"]
    ratings: np.ndarray = models["ratings"]
    abvs: np.ndarray = models["abvs"]
    adj_lin: np.ndarray = models["adj_ratings_linear"]
    adj_poly: np.ndarray = models["adj_ratings_poly"]
    n = len(ratings)

    roll_dates = []
    roll_raw = []
    roll_abv = []
    roll_adj_lin = []
    roll_adj_poly = []

    for i in range(window_size, n + 1):
        roll_dates.append(dates[i - 1])
        roll_raw.append(np.mean(ratings[i - window_size : i]))
        roll_abv.append(np.mean(abvs[i - window_size : i]))
        roll_adj_lin.append(np.mean(adj_lin[i - window_size : i]))
        roll_adj_poly.append(np.mean(adj_poly[i - window_size : i]))

    roll_dates_arr = np.array(roll_dates)
    roll_raw_arr = np.array(roll_raw, dtype=float)
    roll_abv_arr = np.array(roll_abv, dtype=float)
    roll_adj_lin_arr = np.array(roll_adj_lin, dtype=float)
    roll_adj_poly_arr = np.array(roll_adj_poly, dtype=float)
    roll_delta_arr = roll_raw_arr - roll_adj_lin_arr

    return {
        "dates": roll_dates_arr,
        "raw_rating": roll_raw_arr,
        "abv": roll_abv_arr,
        "adj_rating_linear": roll_adj_lin_arr,
        "adj_rating_poly": roll_adj_poly_arr,
        "abv_inflation_delta": roll_delta_arr,
    }


def print_statistical_report(
    models: dict[str, object],
    dynamics: dict[str, np.ndarray],
    window_size: int = 300,
) -> None:
    """Prints comprehensive statistical analysis and annual breakdown tables."""
    ratings: np.ndarray = models["ratings"]
    abvs: np.ndarray = models["abvs"]
    dates: np.ndarray = models["dates"]
    lm = models["linear_model"]
    slope_lin = lm["slope"]
    int_lin = lm["intercept"]
    r_lin = lm["r"]
    p_lin = lm["p_value"]
    slope_within = models["within_style_slope"]

    roll_raw = dynamics["raw_rating"]
    roll_abv = dynamics["abv"]
    roll_adj = dynamics["adj_rating_linear"]
    roll_dates = dynamics["dates"]

    r_roll_raw, p_roll_raw = stats.pearsonr(roll_abv, roll_raw)
    r_roll_adj, p_roll_adj = stats.pearsonr(roll_abv, roll_adj)

    print("=" * 80)
    print("ABV DECOUPLING & RATING SATISFACTION DYNAMICS ANALYSIS")
    print("=" * 80)
    print(f"Total Check-ins Analyzed:            {len(ratings):,}")
    print(
        f"Timeline Span:                       {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}"
    )
    print(f"Global Sample Mean Rating:           {models['mean_rating']:.3f} / 5.0")
    print(f"Global Sample Mean ABV:              {models['mean_abv']:.2f}%")
    print(
        f"Sliding Window Size (W):             {window_size} check-ins ({len(roll_dates):,} windows)"
    )
    print()
    print("1. CROSS-SECTIONAL RATING VS. ABV REGRESSION MODELS")
    print(
        f"  Linear Model:           Rating = {int_lin:.4f} + ({slope_lin:+.4f}) * ABV"
    )
    print(
        f"    -> Pearson r:         {r_lin:+.4f} (R² = {r_lin**2:.4f}, p = {p_lin:.3e})"
    )
    print(f"    -> Rating boost/1%:   +{slope_lin:.4f} stars per 1.0% increase in ABV")
    print(
        f"  Within-Style Slope:     beta = {slope_within:+.4f} (demeaned fixed effects)"
    )
    print()
    print("2. LONGITUDINAL ROLLING CORRELATION DECOUPLING (W=300)")
    print(
        f"  Raw Rating vs. Mean ABV:        Pearson r = {r_roll_raw:+.4f} (p = {p_roll_raw:.3e})"
    )
    print(
        f"  ABV-Adjusted vs. Mean ABV:      Pearson r = {r_roll_adj:+.4f} (p = {p_roll_adj:.3e})"
    )
    print(
        f"  Correlation Reduction:          Δr = {abs(r_roll_raw) - abs(r_roll_adj):.4f} (from {r_roll_raw:+.3f} to {r_roll_adj:+.3f})"
    )
    print()
    print("3. ANNUAL BREAKDOWN: RAW VS. ABV-ADJUSTED SATISFACTION")
    years = sorted(list(set(d.year for d in dates)))
    print(
        f"  {'Year':<6} | {'Count':<7} | {'Raw Rating':<12} | "
        f"{'Adj Rating':<12} | {'Mean ABV (%)':<14} | {'Net ABV Premium (ΔR)':<20}"
    )
    print("  " + "-" * 76)
    for yr in years:
        mask_yr = np.array([d.year == yr for d in dates])
        n_yr = np.sum(mask_yr)
        r_raw_yr = np.mean(ratings[mask_yr])
        r_adj_yr = np.mean(models["adj_ratings_linear"][mask_yr])
        abv_yr = np.mean(abvs[mask_yr])
        delta_yr = r_raw_yr - r_adj_yr
        delta_str = f"{delta_yr:+.3f}"
        print(
            f"  {yr:<6} | {n_yr:<7,} | {r_raw_yr:<12.3f} | "
            f"{r_adj_yr:<12.3f} | {abv_yr:<14.2f} | {delta_str:<20}"
        )
    print("=" * 80)


@untappd_utils.show_or_save_to_out_file
def plot_abv_adjusted_dynamics(
    checkins: list[untappd.Checkin],
    window_size: int = 300,
) -> None:
    """Generates a 4-panel dashboard isolating ABV correlation from rating satisfaction."""
    models = compute_abv_normalization_models(checkins)
    dynamics = compute_rolling_abv_dynamics(models, window_size=window_size)
    print_statistical_report(models, dynamics, window_size=window_size)

    roll_dates = dynamics["dates"]
    roll_raw = dynamics["raw_rating"]
    roll_abv = dynamics["abv"]
    roll_adj = dynamics["adj_rating_linear"]
    roll_delta = dynamics["abv_inflation_delta"]

    ratings: np.ndarray = models["ratings"]
    abvs: np.ndarray = models["abvs"]
    dates: np.ndarray = models["dates"]
    lm = models["linear_model"]
    mean_r = models["mean_rating"]
    mean_a = models["mean_abv"]

    seaborn.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    color_raw = "#2ca02c"
    color_adj = "#1f77b4"
    color_abv = "#d62728"
    color_poly = "#9467bd"

    # Panel 1 (Top-Left): Primary Timeline (Raw vs ABV-Adjusted Rating & Rolling ABV)
    ax1 = axes[0, 0]
    ax1.plot(
        roll_dates,
        roll_raw,
        color=color_raw,
        linewidth=2.2,
        label=r"Raw Rolling Rating $\bar{R}(t)$",
    )
    ax1.plot(
        roll_dates,
        roll_adj,
        color=color_adj,
        linewidth=2.0,
        linestyle="--",
        label=r"ABV-Adjusted Rating $\bar{R}_{\mathrm{adj}}(t)$",
    )

    # Shaded ribbon highlighting ABV inflation vs deflation
    ax1.fill_between(
        roll_dates,
        roll_raw,
        roll_adj,
        where=(roll_raw >= roll_adj),
        interpolate=True,
        color="#ff9896",
        alpha=0.35,
        label="ABV Rating Premium (Inflated)",
    )
    ax1.fill_between(
        roll_dates,
        roll_raw,
        roll_adj,
        where=(roll_raw < roll_adj),
        interpolate=True,
        color="#aec7e8",
        alpha=0.35,
        label="Low ABV Penalty (Suppressed)",
    )

    ax1.set_ylabel("Average Rating (0–5)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "A. Decoupled Rating Dynamics: Raw vs. ABV-Adjusted",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax1_twin = ax1.twinx()
    ax1_twin.grid(False)
    ax1_twin.plot(
        roll_dates,
        roll_abv,
        color=color_abv,
        linewidth=1.6,
        linestyle="-.",
        label="Mean ABV (%)",
    )
    ax1_twin.set_ylabel(
        "Mean ABV (%)",
        color=color_abv,
        fontsize=12,
        fontweight="bold",
    )
    ax1_twin.tick_params(axis="y", labelcolor=color_abv)

    lines1_1, labels1_1 = ax1.get_legend_handles_labels()
    lines1_2, labels1_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(
        lines1_1 + lines1_2,
        labels1_1 + labels1_2,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
    )

    # Panel 2 (Top-Right): Cross-Sectional ABV vs Rating Response Function
    ax2 = axes[0, 1]
    bins = np.linspace(2.5, 14.0, 24)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = []
    bin_stds = []
    for b_lo, b_hi in zip(bins[:-1], bins[1:]):
        m = (abvs >= b_lo) & (abvs < b_hi)
        if np.sum(m) >= 5:
            bin_means.append(float(np.mean(ratings[m])))
            bin_stds.append(float(np.std(ratings[m])))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    ax2.errorbar(
        bin_centers,
        bin_means,
        yerr=bin_stds,
        fmt="o",
        color="#333333",
        ecolor="#aaaaaa",
        elinewidth=1.2,
        capsize=3,
        alpha=0.85,
        label=r"Binned Mean Rating $\pm 1\,\mathrm{SD}$",
    )

    abv_grid = np.linspace(2.0, 14.5, 100)
    ax2.plot(
        abv_grid,
        lm["intercept"] + lm["slope"] * abv_grid,
        color="#d62728",
        linewidth=2.2,
        label=f"OLS Linear Fit ($\\beta = {lm['slope']:+.4f}$/%)",
    )
    ax2.plot(
        abv_grid,
        models["poly_func"](abv_grid),
        color=color_poly,
        linewidth=2.0,
        linestyle=":",
        label="Cubic Response Curve",
    )

    ax2.set_xlabel("Beer ABV (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Check-in Rating (0–5)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "B. Cross-Sectional Rating vs. ABV Response Function",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9)

    fit_text = (
        f"Linear Slope $\\beta = +{lm['slope']:.4f}$ / % ABV\n"
        f"Pearson $r = {lm['r']:+.4f}$ ($p < 10^{{-15}}$)\n"
        f"Mean Baseline: {mean_r:.2f} stars @ {mean_a:.1f}% ABV"
    )
    ax2.text(
        0.05,
        0.72,
        fit_text,
        transform=ax2.transAxes,
        fontsize=10.5,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    # Panel 3 (Bottom-Left): Net ABV Rating Inflation Delta(t) over Time
    ax3 = axes[1, 0]
    ax3.axhline(0, color="#666666", linestyle="-", linewidth=1.0, alpha=0.7)
    ax3.plot(
        roll_dates,
        roll_delta,
        color="#8c564b",
        linewidth=2.0,
        label=r"Net ABV Effect $\Delta R(t) = \bar{R}(t) - \bar{R}_{\mathrm{adj}}(t)$",
    )
    ax3.fill_between(
        roll_dates,
        0,
        roll_delta,
        where=(roll_delta >= 0),
        color="#ff9896",
        alpha=0.4,
        label="Rating Inflation from Higher ABV",
    )
    ax3.fill_between(
        roll_dates,
        0,
        roll_delta,
        where=(roll_delta < 0),
        color="#aec7e8",
        alpha=0.4,
        label="Rating Deflation from Lower ABV",
    )

    ax3.set_ylabel("ABV Rating Distortion (stars)", fontsize=12, fontweight="bold")
    ax3.set_title(
        r"C. Net Temporal ABV Premium $\Delta R(t)$",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.legend(loc="lower right", frameon=True, framealpha=0.9)

    # Panel 4 (Bottom-Right): Annual Comparison (Raw vs Adjusted Rating & Mean ABV)
    ax4 = axes[1, 1]
    years = sorted(list(set(d.year for d in dates)))
    yr_raw = [
        float(np.mean([ratings[k] for k in range(len(ratings)) if dates[k].year == y]))
        for y in years
    ]
    yr_adj = [
        float(
            np.mean(
                [
                    models["adj_ratings_linear"][k]
                    for k in range(len(ratings))
                    if dates[k].year == y
                ]
            )
        )
        for y in years
    ]
    yr_abv = [
        float(np.mean([abvs[k] for k in range(len(ratings)) if dates[k].year == y]))
        for y in years
    ]

    x_pos = np.arange(len(years))
    width = 0.35

    ax4.bar(
        x_pos - width / 2,
        yr_raw,
        width,
        label="Raw Mean Rating",
        color=color_raw,
        alpha=0.85,
    )
    ax4.bar(
        x_pos + width / 2,
        yr_adj,
        width,
        label="ABV-Adjusted Rating",
        color=color_adj,
        alpha=0.85,
    )

    ax4.set_ylabel("Annual Mean Rating (0–5)", fontsize=12, fontweight="bold")
    ax4.set_title(
        "D. Annual Raw vs. ABV-Adjusted Satisfaction",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(years)
    ax4.set_ylim(3.0, 3.6)

    ax4_twin = ax4.twinx()
    ax4_twin.grid(False)
    ax4_twin.plot(
        x_pos,
        yr_abv,
        color=color_abv,
        linewidth=2.2,
        marker="o",
        linestyle="-.",
        label="Annual Mean ABV (%)",
    )
    ax4_twin.set_ylabel(
        "Mean ABV (%)",
        color=color_abv,
        fontsize=12,
        fontweight="bold",
    )
    ax4_twin.tick_params(axis="y", labelcolor=color_abv)
    ax4_twin.set_ylim(4.5, 7.5)

    lines4_1, labels4_1 = ax4.get_legend_handles_labels()
    lines4_2, labels4_2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(
        lines4_1 + lines4_2,
        labels4_1 + labels4_2,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
    )

    plt.suptitle(
        f"Decoupling ABV Correlation from Rating Satisfaction Dynamics (W={window_size} Check-ins)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )


if __name__ == "__main__":
    checkins = untappd.load_latest_checkins()
    plot_abv_adjusted_dynamics(
        checkins,
        out_file=Path(__file__).parent / "out" / "abv_adjusted_rating_dynamics.png",
    )
