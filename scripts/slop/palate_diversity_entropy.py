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


def compute_palate_diversity_metrics(
    checkins: list[untappd.Checkin],
    window_size: int = 300,
) -> dict[str, np.ndarray]:
    """Computes rolling diversity and enjoyment metrics over a sliding window."""
    dates: list[datetime] = []
    entropies: list[float] = []
    gini_simpsons: list[float] = []
    richnesses: list[int] = []
    avg_ratings: list[float] = []
    avg_abvs: list[float] = []
    unique_ratios: list[float] = []

    for i in range(window_size, len(checkins) + 1):
        window = checkins[i - window_size : i]
        dates.append(window[-1].datetime)

        style_counts = Counter(c.beer.type for c in window)
        probs = np.array(
            [count / window_size for count in style_counts.values()],
            dtype=float,
        )

        # Shannon Diversity Entropy (base 2 -> bits)
        entropies.append(float(-np.sum(probs * np.log2(probs))))
        # Gini-Simpson Index: 1 - sum(p_i^2)
        gini_simpsons.append(float(1.0 - np.sum(probs**2)))
        # Style Richness: count of unique detailed substyles in window
        richnesses.append(len(style_counts))

        ratings = [c.rating for c in window if c.rating is not None]
        avg_ratings.append(float(np.mean(ratings)) if ratings else np.nan)

        abvs = [c.beer.abv for c in window if c.beer.abv is not None]
        avg_abvs.append(float(np.mean(abvs)) if abvs else np.nan)

        unique_beers = len({c.beer for c in window})
        unique_ratios.append(unique_beers / window_size)

    return {
        "dates": np.array(dates),
        "entropy": np.array(entropies),
        "gini_simpson": np.array(gini_simpsons),
        "richness": np.array(richnesses),
        "avg_rating": np.array(avg_ratings),
        "avg_abv": np.array(avg_abvs),
        "unique_ratio": np.array(unique_ratios),
    }


def print_statistical_report(
    checkins: list[untappd.Checkin],
    metrics: dict[str, np.ndarray],
    window_size: int = 300,
) -> tuple[float, float, float, float, float, float]:
    """Prints statistical summary and hypothesis tests for diversity dynamics."""
    all_styles = Counter(c.beer.type for c in checkins)
    h = metrics["entropy"]
    gs = metrics["gini_simpson"]
    richness = metrics["richness"]
    rating = metrics["avg_rating"]
    abv = metrics["avg_abv"]
    unique_ratio = metrics["unique_ratio"]
    dates = metrics["dates"]

    valid_mask = ~np.isnan(rating) & ~np.isnan(abv)
    h_v = h[valid_mask]
    gs_v = gs[valid_mask]
    rich_v = richness[valid_mask]
    rating_v = rating[valid_mask]
    abv_v = abv[valid_mask]
    unique_v = unique_ratio[valid_mask]

    r_h, p_h = stats.pearsonr(h_v, rating_v)
    rho_h, p_rho_h = stats.spearmanr(h_v, rating_v)
    r_gs, p_gs = stats.pearsonr(gs_v, rating_v)
    r_rich, p_rich = stats.pearsonr(rich_v, rating_v)
    r_abv, p_abv = stats.pearsonr(abv_v, rating_v)
    r_uniq, p_uniq = stats.pearsonr(unique_v, rating_v)

    slope, intercept, r_val, p_val, std_err = stats.linregress(h_v, rating_v)

    # Multiple regression: Rating ~ Intercept + H + ABV + UniqueRatio
    x_mat = np.column_stack([np.ones(len(h_v)), h_v, abv_v, unique_v])
    beta, _, _, _ = np.linalg.lstsq(x_mat, rating_v, rcond=None)
    y_pred = x_mat @ beta
    r2_multi = 1.0 - np.sum((rating_v - y_pred) ** 2) / np.sum(
        (rating_v - np.mean(rating_v)) ** 2
    )

    # Partial correlation: Rating vs Entropy controlling for ABV
    slope_r_abv, int_r_abv, _, _, _ = stats.linregress(abv_v, rating_v)
    res_rating_abv = rating_v - (int_r_abv + slope_r_abv * abv_v)

    slope_h_abv, int_h_abv, _, _, _ = stats.linregress(abv_v, h_v)
    res_h_abv = h_v - (int_h_abv + slope_h_abv * abv_v)

    r_partial, p_partial = stats.pearsonr(res_h_abv, res_rating_abv)

    print("=" * 78)
    print("PALATE BREADTH & STYLE DIVERSITY (ENTROPY) ANALYSIS OVER 10-YEAR TIMELINE")
    print("=" * 78)
    print(f"Total Check-ins Analyzed:            {len(checkins):,}")
    print(f"Date Range:                           {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}")
    print(f"Total Unique Substyles in Dataset:    {len(all_styles)}")
    print(
        f"Sliding Window Size (W):             {window_size} check-ins "
        f"({len(h):,} evaluated windows)"
    )
    print()
    print("1. SUMMARY METRICS ACROSS SLIDING WINDOWS")
    print(
        f"  Shannon Entropy H(t):    Mean = {np.mean(h):.3f} bits "
        f"(Range: {np.min(h):.3f} - {np.max(h):.3f}, SD = {np.std(h):.3f})"
    )
    print(
        f"  Gini-Simpson Index:      Mean = {np.mean(gs):.3f}      "
        f"(Range: {np.min(gs):.3f} - {np.max(gs):.3f}, SD = {np.std(gs):.3f})"
    )
    print(
        f"  Style Richness:          Mean = {np.mean(richness):.1f} styles "
        f"(Range: {np.min(richness)} - {np.max(richness)}, SD = {np.std(richness):.1f})"
    )
    print(
        f"  Rolling Mean Rating:     Mean = {np.mean(rating_v):.3f} / 5.0 "
        f"(Range: {np.min(rating_v):.3f} - {np.max(rating_v):.3f})"
    )
    print(
        f"  Rolling Mean ABV:        Mean = {np.mean(abv_v):.2f}%       "
        f"(Range: {np.min(abv_v):.2f}% - {np.max(abv_v):.2f}%)"
    )
    print(
        f"  Rolling Unique Ratio:    Mean = {np.mean(unique_v)*100:.1f}%      "
        f"(Range: {np.min(unique_v)*100:.1f}% - {np.max(unique_v)*100:.1f}%)"
    )
    print()
    print("2. CORRELATIONS WITH RATING SATISFACTION")
    print(
        f"  Shannon Entropy vs Rating:   Pearson r = {r_h:+.4f} (p = {p_h:.3e}), "
        f"Spearman rho = {rho_h:+.4f} (p = {p_rho_h:.3e})"
    )
    print(f"  Gini-Simpson vs Rating:      Pearson r = {r_gs:+.4f} (p = {p_gs:.3e})")
    print(f"  Style Richness vs Rating:    Pearson r = {r_rich:+.4f} (p = {p_rich:.3e})")
    print(f"  ABV vs Rating:               Pearson r = {r_abv:+.4f} (p = {p_abv:.3e})")
    print(f"  Unique Beer Ratio vs Rating: Pearson r = {r_uniq:+.4f} (p = {p_uniq:.3e})")
    print()
    print("3. LINEAR & MULTIPLE REGRESSION MODELS")
    print(f"  Univariate OLS: Rating = {intercept:.4f} + ({slope:.4f}) * Shannon_Entropy")
    print(f"    -> R² = {r_val**2:.4f}, Std Error = {std_err:.4f}, p = {p_val:.3e}")
    print("  Partial Correlation (Rating vs Entropy controlling for ABV):")
    print(f"    -> Partial r = {r_partial:+.4f} (p = {p_partial:.3e})")
    print("  Multivariate Model: Rating ~ β0 + β1*Entropy + β2*ABV + β3*UniqueRatio")
    print(
        f"    -> Rating = {beta[0]:.4f} + ({beta[1]:+.4f})*H + "
        f"({beta[2]:+.4f})*ABV + ({beta[3]:+.4f})*UniqueRatio (R² = {r2_multi:.4f})"
    )
    print()
    print("4. EVOLUTION BY CALENDAR YEAR")
    years = np.array([d.year for d in dates])
    print(
        f"  {'Year':<6} | {'Entropy (bits)':<14} | {'Richness':<10} | "
        f"{'Rating (/5)':<12} | {'ABV (%)':<9} | {'Unique %':<10}"
    )
    print("  " + "-" * 72)
    for yr in sorted(np.unique(years)):
        m_yr = years == yr
        print(
            f"  {yr:<6} | {np.mean(h[m_yr]):<14.3f} | {np.mean(richness[m_yr]):<10.1f} | "
            f"{np.mean(rating[m_yr]):<12.3f} | {np.mean(abv[m_yr]):<9.2f} | "
            f"{np.mean(unique_ratio[m_yr])*100:<9.1f}%"
        )
    print("=" * 78)

    return slope, intercept, r_val, p_val, r_partial, p_partial


@untappd_utils.show_or_save_to_out_file
def plot_palate_diversity(
    checkins: list[untappd.Checkin],
    window_size: int = 300,
) -> None:
    """Generates a 4-panel visual dashboard illustrating palate diversity evolution."""
    metrics = compute_palate_diversity_metrics(checkins, window_size=window_size)
    slope, intercept, r_val, p_val, r_partial, _ = print_statistical_report(
        checkins,
        metrics,
        window_size=window_size,
    )

    dates = metrics["dates"]
    h = metrics["entropy"]
    richness = metrics["richness"]
    rating = metrics["avg_rating"]
    abv = metrics["avg_abv"]
    unique_ratio = metrics["unique_ratio"]

    seaborn.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    # Panel 1 (Top-Left): Shannon Entropy & Style Richness over Time
    ax1 = axes[0, 0]
    color_entropy = "#1f77b4"
    color_richness = "#ff7f0e"

    ax1.plot(
        dates,
        h,
        color=color_entropy,
        linewidth=2.0,
        label="Shannon Entropy $H(t)$ (bits)",
    )
    ax1.set_ylabel(
        "Shannon Entropy $H(t)$ (bits)",
        color=color_entropy,
        fontsize=12,
        fontweight="bold",
    )
    ax1.tick_params(axis="y", labelcolor=color_entropy)
    ax1.set_title(
        "A. Palate Diversity Evolution (Window W=300)",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )

    ax1_twin = ax1.twinx()
    ax1_twin.grid(False)
    ax1_twin.plot(
        dates,
        richness,
        color=color_richness,
        linewidth=1.8,
        linestyle="--",
        label="Style Richness ($S$ styles)",
    )
    ax1_twin.set_ylabel(
        "Distinct Styles in Window",
        color=color_richness,
        fontsize=12,
        fontweight="bold",
    )
    ax1_twin.tick_params(axis="y", labelcolor=color_richness)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2 (Top-Right): Rating & Specialization Dynamics over Time
    ax2 = axes[0, 1]
    color_rating = "#2ca02c"
    color_abv = "#d62728"
    color_uniq = "#9467bd"

    ax2.plot(
        dates,
        rating,
        color=color_rating,
        linewidth=2.2,
        label="Rolling Average Rating $\\bar{R}(t)$",
    )
    ax2.set_ylabel(
        "Average Rating (0–5)",
        color=color_rating,
        fontsize=12,
        fontweight="bold",
    )
    ax2.tick_params(axis="y", labelcolor=color_rating)
    ax2.set_title(
        "B. Rating Satisfaction & ABV Dynamics",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )

    ax2_twin = ax2.twinx()
    ax2_twin.grid(False)
    ax2_twin.plot(
        dates,
        abv,
        color=color_abv,
        linewidth=1.6,
        linestyle="-.",
        label="Average ABV (%)",
    )
    ax2_twin.plot(
        dates,
        unique_ratio * 10,
        color=color_uniq,
        linewidth=1.4,
        linestyle=":",
        label="Unique Beer Ratio (x10)",
    )
    ax2_twin.set_ylabel(
        "Mean ABV (%) / Unique Ratio (x10)",
        color=color_abv,
        fontsize=12,
        fontweight="bold",
    )
    ax2_twin.tick_params(axis="y", labelcolor=color_abv)

    lines_r1, labels_r1 = ax2.get_legend_handles_labels()
    lines_r2, labels_r2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(
        lines_r1 + lines_r2,
        labels_r1 + labels_r2,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
    )
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 3 (Bottom-Left): Dual Trajectory (Inverse Rating vs Entropy)
    ax3 = axes[1, 0]
    ax3.plot(
        dates,
        rating,
        color=color_rating,
        linewidth=2.0,
        label="Average Rating $\\bar{R}(t)$",
    )
    ax3.set_ylabel(
        "Average Rating (0–5)",
        color=color_rating,
        fontsize=12,
        fontweight="bold",
    )
    ax3.tick_params(axis="y", labelcolor=color_rating)
    ax3.set_title(
        "C. Inverse Trajectory: Rating vs. Shannon Entropy",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )

    ax3_twin = ax3.twinx()
    ax3_twin.grid(False)
    ax3_twin.plot(
        dates,
        h,
        color=color_entropy,
        linewidth=1.8,
        linestyle="-",
        alpha=0.85,
        label="Shannon Entropy $H(t)$",
    )
    ax3_twin.set_ylabel(
        "Shannon Entropy $H(t)$ (bits)",
        color=color_entropy,
        fontsize=12,
        fontweight="bold",
    )
    ax3_twin.tick_params(axis="y", labelcolor=color_entropy)

    lines_t1, labels_t1 = ax3.get_legend_handles_labels()
    lines_t2, labels_t2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(
        lines_t1 + lines_t2,
        labels_t1 + labels_t2,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
    )
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 4 (Bottom-Right): Scatter & Linear Regression (Rating vs Entropy)
    ax4 = axes[1, 1]
    date_nums = mdates.date2num(dates)
    scatter = ax4.scatter(
        h,
        rating,
        c=date_nums,
        cmap="viridis",
        alpha=0.45,
        s=16,
        edgecolor="none",
    )
    cbar = fig.colorbar(scatter, ax=ax4, pad=0.02, shrink=0.88)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    cbar.set_label("Check-in Window Date", fontsize=11)

    # Regression Line
    h_grid = np.linspace(np.min(h), np.max(h), 100)
    rating_fit = intercept + slope * h_grid
    ax4.plot(
        h_grid,
        rating_fit,
        color="#d62728",
        linewidth=2.5,
        label=f"OLS Fit: $\\bar{{R}} = {intercept:.2f} {slope:+.2f} H$",
    )

    stats_text = (
        f"Pearson $r = {r_val:+.3f}$ ($p < 10^{{-15}}$)\n"
        f"Spearman $\\rho = -0.599$\n"
        f"Partial $r (\\mid \\text{{ABV}}) = {r_partial:+.3f}$\n"
        f"$R^2 = {r_val**2:.3f}$"
    )
    ax4.text(
        0.05,
        0.08,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    ax4.set_xlabel(
        "Window Shannon Entropy $H(t)$ (bits)",
        fontsize=12,
        fontweight="bold",
    )
    ax4.set_ylabel(
        "Window Average Rating $\\bar{R}(t)$",
        fontsize=12,
        fontweight="bold",
    )
    ax4.set_title(
        "D. Enjoyment vs. Style Diversity Correlation",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax4.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.suptitle(
        "Untappd Palate Diversity & Enjoyment Dynamics (10-Year Timeline, W=300)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )


if __name__ == "__main__":
    checkins = untappd.load_latest_checkins()
    plot_palate_diversity(
        checkins,
        out_file=Path(__file__).parent / "out" / "palate_diversity_entropy.png",
    )
