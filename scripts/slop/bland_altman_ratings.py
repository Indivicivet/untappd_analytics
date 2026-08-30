from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn

import untappd
import untappd_utils


@untappd_utils.show_or_save_to_out_file
def plot_bland_altman(checkins):
    valid_checkins = [
        c
        for c in checkins
        if c.rating is not None
        and c.beer.global_rating is not None
        and c.beer.global_rating > 0
    ]

    personal = np.array([c.rating for c in valid_checkins])
    global_scores = np.array([c.beer.global_rating for c in valid_checkins])
    styles = [c.beer.get_style_category() for c in valid_checkins]

    means = (personal + global_scores) / 2.0
    diffs = personal - global_scores  # personal - global
    n = len(diffs)

    # Core Bland-Altman statistics
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # Standard errors for confidence intervals
    se_mean = std_diff / np.sqrt(n)
    ci_mean = (mean_diff - 1.96 * se_mean, mean_diff + 1.96 * se_mean)
    se_loa = np.sqrt(3.0 * (std_diff**2) / n)
    ci_loa_upper = (loa_upper - 1.96 * se_loa, loa_upper + 1.96 * se_loa)
    ci_loa_lower = (loa_lower - 1.96 * se_loa, loa_lower + 1.96 * se_loa)

    # Test for proportional bias (slope of diffs vs means)
    slope, intercept, r_val, p_val, std_err = stats.linregress(means, diffs)

    # Print statistical report
    print("=" * 65)
    print("BLAND-ALTMAN AGREEMENT ANALYSIS (Personal vs. Global Untappd)")
    print("=" * 65)
    print(f"Total check-ins analyzed:       {n}")
    print(
        f"Mean Difference (Personal - Global Bias): {mean_diff:+.4f} (95% CI: {ci_mean[0]:+.4f} to {ci_mean[1]:+.4f})"
    )
    print(f"Standard Deviation of Diffs:    {std_diff:.4f}")
    print(
        f"Upper Limit of Agreement (+1.96 SD):     {loa_upper:+.4f} (95% CI: {ci_loa_upper[0]:+.4f} to {ci_loa_upper[1]:+.4f})"
    )
    print(
        f"Lower Limit of Agreement (-1.96 SD):     {loa_lower:+.4f} (95% CI: {ci_loa_lower[0]:+.4f} to {ci_loa_lower[1]:+.4f})"
    )
    print(f"\nProportional Bias Test (OLS Regression of Diffs on Means):")
    print(f"  Slope (beta):  {slope:+.4f} (SE: {std_err:.4f})")
    print(f"  Intercept:  {intercept:+.4f}")
    print(f"  R-squared:  {r_val**2:.4f}")
    print(f"  p-value:    {p_val:.4g}")
    if p_val < 0.05:
        if slope > 0:
            print(
                "  Conclusion: Significant positive proportional bias detected (p < 0.05)."
            )
            print(
                "              You rate high-end beers even higher than the crowd, and low-end beers lower."
            )
        else:
            print(
                "  Conclusion: Significant negative proportional bias detected (p < 0.05)."
            )
            print(
                "              Your rating scale is compressed relative to the crowd."
            )
    else:
        print(
            "  Conclusion: No significant proportional bias detected (p >= 0.05). Agreement is uniform."
        )

    # Breakdown by style category
    style_diffs = defaultdict(list)
    for s, d in zip(styles, diffs):
        style_diffs[s].append(d)

    print("\n" + "-" * 65)
    print(
        f"{'Style Category':<15} | {'N':<6} | {'Mean Bias':<10} | {'SD':<8} | {'95% LoA Interval':<20}"
    )
    print("-" * 65)
    for cat in sorted(style_diffs, key=lambda c: len(style_diffs[c]), reverse=True):
        arr = np.array(style_diffs[cat])
        m = np.mean(arr)
        s = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        print(
            f"{cat:<15} | {len(arr):<6d} | {m:>+9.3f}  | {s:>7.3f} | [{m - 1.96 * s:>+6.2f}, {m + 1.96 * s:>+6.2f}]"
        )
    print("=" * 65)

    # Plotting
    seaborn.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: Bland-Altman Scatter
    # Add minor jitter to discrete personal ratings for visual density clarity
    jitter_m = means + np.random.normal(0, 0.015, size=n)
    jitter_d = diffs + np.random.normal(0, 0.015, size=n)

    ax1.scatter(
        jitter_m, jitter_d, alpha=0.15, color="#1f77b4", s=18, edgecolors="none"
    )
    ax1.axhline(
        mean_diff,
        color="red",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean Bias ({mean_diff:+.2f})",
    )
    ax1.axhline(
        loa_upper,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Upper LoA (+1.96 SD: {loa_upper:+.2f})",
    )
    ax1.axhline(
        loa_lower,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Lower LoA (-1.96 SD: {loa_lower:+.2f})",
    )

    # Shaded LoA zone
    ax1.axhspan(loa_lower, loa_upper, color="gray", alpha=0.08)

    # Regression trend line
    x_vals = np.linspace(min(means), max(means), 100)
    ax1.plot(
        x_vals,
        intercept + slope * x_vals,
        color="darkblue",
        linestyle=":",
        linewidth=2,
        label=f"Proportional Trend (slope: {slope:+.2f}, p={p_val:.2g})",
    )

    ax1.set_xlabel("Mean Rating: (Personal + Global) / 2", fontsize=12)
    ax1.set_ylabel("Difference: Personal - Global", fontsize=12)
    ax1.set_title("Bland-Altman Agreement Plot", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)

    # Right plot: Style Category Forest Plot (Bias and 95% LoA)
    cats_to_plot = [
        c
        for c in sorted(style_diffs, key=lambda c: len(style_diffs[c]), reverse=True)
        if len(style_diffs[c]) >= 10
    ]
    y_pos = np.arange(len(cats_to_plot))
    cat_means = [np.mean(style_diffs[c]) for c in cats_to_plot]
    cat_sds = [np.std(style_diffs[c], ddof=1) for c in cats_to_plot]
    cat_loas_low = [m - 1.96 * s for m, s in zip(cat_means, cat_sds)]
    cat_loas_high = [m + 1.96 * s for m, s in zip(cat_means, cat_sds)]

    ax2.axvline(0, color="gray", linestyle="-", linewidth=1)
    ax2.axvline(
        mean_diff,
        color="red",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
        label=f"Overall Bias ({mean_diff:+.2f})",
    )

    for i, (m, low, high, c) in enumerate(
        zip(cat_means, cat_loas_low, cat_loas_high, cats_to_plot)
    ):
        ax2.plot([low, high], [i, i], color="#2b5c8f", linewidth=2.5)
        ax2.plot(m, i, "o", color="#d62728", markersize=7)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"{c} (N={len(style_diffs[c])})" for c in cats_to_plot], fontsize=11
    )
    ax2.set_xlabel("Personal - Global Difference (Mean +/- 1.96 SD)", fontsize=12)
    ax2.set_title(
        "Agreement & Limits of Agreement by Style", fontsize=14, fontweight="bold"
    )
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()


if __name__ == "__main__":
    checkins = untappd.load_latest_checkins()
    plot_bland_altman(
        checkins,
        out_file=Path(__file__).parent / "out" / "bland_altman_agreement.png",
    )
