from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn

import untappd
import untappd_utils

# Ensure UTF-8 stdout encoding on Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def natural_cubic_spline_basis(
    x: np.ndarray,
    knots: Sequence[float],
) -> np.ndarray:
    """
    Constructs a Natural Cubic Spline (NCS) basis matrix for 1D input x.
    Guarantees linearity beyond boundary knots (knots[0] and knots[-1]).
    Returns basis matrix of shape (len(x), len(knots) - 1).
    """
    knots = np.asarray(knots, dtype=float)
    x = np.asarray(x, dtype=float)
    k_K = knots[-1]
    k_Km1 = knots[-2]

    basis = [x]

    def d_k(k):
        term1 = np.maximum(0.0, x - k) ** 3
        term2 = np.maximum(0.0, x - k_K) ** 3
        return (term1 - term2) / (k_K - k)

    d_Km1 = d_k(k_Km1)
    for k in knots[:-2]:
        basis.append(d_k(k) - d_Km1)

    return np.column_stack(basis)


@dataclass
class MultivariateRatingModel:
    global_mean: float
    beta_global: float
    style_categories: list[str]
    style_means: dict[str, float]
    style_adjustments: dict[str, float]
    abv_knots: np.ndarray
    abv_basis_mean: np.ndarray
    beta_abv: np.ndarray
    time_knots: np.ndarray
    time_basis_mean: np.ndarray
    beta_time: np.ndarray
    baseline_rating: float
    r_squared: float
    sigma_epsilon: float
    n_samples: int
    n_features: int

    def predict_abv_partial_effect(self, abv_vals: np.ndarray) -> np.ndarray:
        basis = natural_cubic_spline_basis(abv_vals, self.abv_knots)
        centered_basis = basis - self.abv_basis_mean
        return centered_basis @ self.beta_abv

    def predict_time_partial_effect(self, time_days: np.ndarray) -> np.ndarray:
        basis = natural_cubic_spline_basis(time_days, self.time_knots)
        centered_basis = basis - self.time_basis_mean
        return centered_basis @ self.beta_time

    def predict(
        self,
        global_ratings: np.ndarray,
        abv_vals: np.ndarray,
        styles: list[str],
        time_days: np.ndarray,
    ) -> np.ndarray:
        n = len(global_ratings)
        pred = np.full(n, self.baseline_rating, dtype=float)

        # Global rating effect
        pred += self.beta_global * (global_ratings - self.global_mean)

        # Style effect
        style_deltas = np.array(
            [self.style_adjustments.get(s, 0.0) for s in styles],
            dtype=float,
        )
        pred += style_deltas

        # ABV effect
        pred += self.predict_abv_partial_effect(abv_vals)

        # Time drift effect
        pred += self.predict_time_partial_effect(time_days)

        return pred


def fit_multivariate_model(
    checkins: list[untappd.Checkin],
) -> tuple[MultivariateRatingModel, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a multivariate linear model for personal rating R_i using:
    - Global Untappd rating
    - Continuous ABV with natural cubic splines
    - Style category (one-hot categorical)
    - Longitudinal time drift with natural cubic splines

    Returns:
        (model, fitted_ratings, residuals, normalized_residuals)
    """
    ratings = np.array([c.rating for c in checkins], dtype=float)
    global_ratings = np.array([c.beer.global_rating for c in checkins], dtype=float)
    abvs = np.array([c.beer.abv for c in checkins], dtype=float)
    styles = [c.beer.get_style_category() for c in checkins]

    min_dt = min(c.datetime for c in checkins)
    time_days = np.array(
        [(c.datetime - min_dt).total_seconds() / 86400.0 for c in checkins],
        dtype=float,
    )

    n = len(checkins)

    # 1. Global rating centered
    global_mean = float(np.mean(global_ratings))
    x_global = (global_ratings - global_mean).reshape(-1, 1)

    # 2. ABV Natural Cubic Spline (knots at 5th, 25th, 50th, 75th, 95th percentiles)
    abv_knots = np.percentile(abvs, [5, 25, 50, 75, 95])
    abv_basis = natural_cubic_spline_basis(abvs, abv_knots)
    abv_basis_mean = np.mean(abv_basis, axis=0)
    x_abv = abv_basis - abv_basis_mean

    # 3. Style Category One-Hot Encoding
    style_categories = sorted(list(set(styles)))
    x_styles = np.zeros((n, len(style_categories)), dtype=float)
    for i, s in enumerate(styles):
        x_styles[i, style_categories.index(s)] = 1.0

    # 4. Longitudinal Time Drift Natural Cubic Spline (5 percentile knots)
    time_knots = np.percentile(time_days, [5, 25, 50, 75, 95])
    time_basis = natural_cubic_spline_basis(time_days, time_knots)
    time_basis_mean = np.mean(time_basis, axis=0)
    x_time = time_basis - time_basis_mean

    # Assemble design matrix: [X_styles, X_global, X_abv, X_time]
    # Note: sum(X_styles) = 1 (intercept is fully spanned by styles)
    X = np.column_stack([x_styles, x_global, x_abv, x_time])

    # OLS estimation via least squares
    coefs, residuals_sum, rank, s = np.linalg.lstsq(X, ratings, rcond=None)

    n_styles = len(style_categories)
    style_coefs = coefs[:n_styles]
    beta_global = float(coefs[n_styles])

    n_abv_cols = x_abv.shape[1]
    beta_abv = coefs[n_styles + 1 : n_styles + 1 + n_abv_cols]

    beta_time = coefs[n_styles + 1 + n_abv_cols :]

    # Compute baseline rating and centered style adjustments
    baseline_rating = float(np.mean(ratings))
    style_means = {cat: float(style_coefs[i]) for i, cat in enumerate(style_categories)}
    style_adjustments = {
        cat: style_means[cat] - baseline_rating for cat in style_categories
    }

    # Model predictions and residuals
    fitted = X @ coefs
    residuals = ratings - fitted

    n_features = X.shape[1]
    df_resid = max(1, n - n_features)
    sigma_epsilon = float(np.sqrt(np.sum(residuals**2) / df_resid))

    ss_total = np.sum((ratings - np.mean(ratings)) ** 2)
    ss_resid = np.sum(residuals**2)
    r_squared = float(1.0 - (ss_resid / ss_total)) if ss_total > 0 else 0.0

    z_residuals = residuals / sigma_epsilon

    model = MultivariateRatingModel(
        global_mean=global_mean,
        beta_global=beta_global,
        style_categories=style_categories,
        style_means=style_means,
        style_adjustments=style_adjustments,
        abv_knots=abv_knots,
        abv_basis_mean=abv_basis_mean,
        beta_abv=beta_abv,
        time_knots=time_knots,
        time_basis_mean=time_basis_mean,
        beta_time=beta_time,
        baseline_rating=baseline_rating,
        r_squared=r_squared,
        sigma_epsilon=sigma_epsilon,
        n_samples=n,
        n_features=n_features,
    )

    return model, fitted, residuals, z_residuals


@untappd_utils.show_or_save_to_out_file
def plot_multivariate_decomposition(
    checkins: list[untappd.Checkin],
    model: MultivariateRatingModel,
    z_residuals: np.ndarray,
):
    """
    Renders a 4-panel diagnostic and factor decomposition figure:
    1. Partial effect of ABV (isolated personal ABV curve)
    2. Style category adjustments (bar chart)
    3. Palate drift over time (longitudinal baseline curve)
    4. Distribution of normalized residuals (histogram & Gaussian fit)
    """
    seaborn.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#fafafa")

    # -------------------------------------------------------------
    # Panel 1: Partial Effect of ABV
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    abvs = np.array([c.beer.abv for c in checkins])
    max_abv_plot = min(18.0, float(np.percentile(abvs, 99.8)))
    grid_abv = np.linspace(0.0, max_abv_plot, 300)
    partial_abv = model.predict_abv_partial_effect(grid_abv)

    ax1.plot(
        grid_abv,
        partial_abv,
        color="#1f77b4",
        linewidth=2.8,
        label="Spline Fit f(ABV)",
    )
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    # Add rug plot of ABV values
    rug_abvs = abvs[abvs <= max_abv_plot]
    ax1.plot(
        rug_abvs,
        np.full_like(rug_abvs, np.min(partial_abv) - 0.04),
        "|",
        color="#1f77b4",
        alpha=0.35,
        markersize=6,
        label="Check-in ABVs",
    )

    ax1.set_xlim(0, max_abv_plot + 0.5)
    ax1.set_title(
        "Partial Effect of ABV on Rating",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("Alcohol By Volume (ABV %)", fontsize=11)
    ax1.set_ylabel("Rating Adjustment (vs. Personal Mean)", fontsize=11)
    ax1.legend(loc="lower right", frameon=True)

    # -------------------------------------------------------------
    # Panel 2: Style Category Adjustments
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    sorted_styles = sorted(
        model.style_categories,
        key=lambda s: model.style_adjustments[s],
        reverse=True,
    )
    style_counts = defaultdict(int)
    for c in checkins:
        style_counts[c.beer.get_style_category()] += 1

    adj_vals = [model.style_adjustments[s] for s in sorted_styles]
    y_pos = np.arange(len(sorted_styles))
    colors = ["#2ca02c" if val >= 0 else "#d62728" for val in adj_vals]

    bars = ax2.barh(
        y_pos, adj_vals, color=colors, alpha=0.85, edgecolor="#333333", height=0.6
    )
    ax2.axvline(0, color="black", linestyle="-", linewidth=1.0)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"{s.title()} (N={style_counts[s]})" for s in sorted_styles],
        fontsize=10.5,
    )
    ax2.invert_yaxis()

    min_val = min(adj_vals)
    max_val = max(adj_vals)
    span = max_val - min_val
    ax2.set_xlim(min_val - 0.25 * span, max_val + 0.25 * span)

    # Add numerical labels
    for bar, val in zip(bars, adj_vals):
        x_offset = 0.008 if val >= 0 else -0.008
        ha = "left" if val >= 0 else "right"
        ax2.text(
            val + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

    ax2.set_title(
        "Style Category Rating Adjustments",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlabel("Rating Adjustment (vs. Personal Mean)", fontsize=11)

    # -------------------------------------------------------------
    # Panel 3: Longitudinal Palate Drift Over Time
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    min_dt = min(c.datetime for c in checkins)
    max_dt = max(c.datetime for c in checkins)
    total_days = (max_dt - min_dt).total_seconds() / 86400.0

    grid_days = np.linspace(0, total_days, 400)
    grid_dts = [
        min_dt + np.timedelta64(int(d * 86400), "s").astype(datetime) for d in grid_days
    ]
    partial_time = model.predict_time_partial_effect(grid_days)

    ax3.plot(
        grid_dts,
        partial_time,
        color="#8c564b",
        linewidth=2.8,
        label="Palate Drift g(Time)",
    )
    ax3.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    ax3.set_title(
        "Longitudinal Palate Drift Over Time",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax3.set_xlabel("Check-in Date", fontsize=11)
    ax3.set_ylabel("Baseline Rating Shift", fontsize=11)
    ax3.legend(loc="upper left", frameon=True)

    # -------------------------------------------------------------
    # Panel 4: Distribution of Normalized Residuals (Z_i)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    n_bins = 45
    counts, bin_edges, _ = ax4.hist(
        z_residuals,
        bins=n_bins,
        density=True,
        color="#4c72b0",
        alpha=0.65,
        edgecolor="#ffffff",
        label="Empirical Z-Residuals",
    )

    z_grid = np.linspace(-4.0, 4.0, 300)
    ax4.plot(
        z_grid,
        stats.norm.pdf(z_grid, 0, 1),
        color="#c44e52",
        linewidth=2.4,
        linestyle="--",
        label="Standard Normal N(0, 1)",
    )

    ax4.axvline(0, color="black", linestyle=":", linewidth=1.0)
    ax4.axvline(
        1.96,
        color="#c44e52",
        linestyle=":",
        linewidth=1.2,
        label="95% Bounds (+/- 1.96)",
    )
    ax4.axvline(-1.96, color="#c44e52", linestyle=":", linewidth=1.2)

    skew = stats.skew(z_residuals)
    kurt = stats.kurtosis(z_residuals)
    in_2sd = np.mean(np.abs(z_residuals) <= 1.96) * 100.0

    stats_text = (
        f"N = {len(z_residuals)}\n"
        f"R² = {model.r_squared:.3f}\n"
        f"σ_ε = {model.sigma_epsilon:.3f}\n"
        f"Skew = {skew:+.2f}\n"
        f"Kurt = {kurt:+.2f}\n"
        f"Within ±2σ: {in_2sd:.1f}%"
    )
    ax4.text(
        0.04,
        0.95,
        stats_text,
        transform=ax4.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#ffffff",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    ax4.set_title(
        "Normalized Rating Residuals Distribution (Z)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax4.set_xlabel("Standardized Residual Z = (R - R̂) / σ_ε", fontsize=11)
    ax4.set_ylabel("Probability Density", fontsize=11)
    ax4.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "Multivariate Rating Normalization & Factor Decomposition",
        fontsize=16,
        fontweight="bold",
        y=0.995,
        color="#1a1a1a",
    )
    plt.tight_layout()


def print_rankings_and_report(
    checkins: list[untappd.Checkin],
    model: MultivariateRatingModel,
    fitted: np.ndarray,
    residuals: np.ndarray,
    z_residuals: np.ndarray,
    top_n: int = 25,
    min_brewery_checkins: int = 5,
):
    print("=" * 110)
    print("                MULTIVARIATE RATING NORMALIZATION MODEL SUMMARY")
    print("=" * 110)
    print(f"Total check-ins analyzed:     {model.n_samples}")
    print(f"Degrees of freedom (residual): {model.n_samples - model.n_features}")
    print(f"Baseline personal mean (R̄):   {model.baseline_rating:.3f}")
    print(f"Global rating mean (Ḡ):       {model.global_mean:.3f}")
    print(f"Global rating slope (β_glob): {model.beta_global:+.3f}")
    print(f"Model R-squared (R²):         {model.r_squared:.4f}")
    print(f"Residual Std Dev (σ_ε):       {model.sigma_epsilon:.4f}")
    print()

    print("Style Category Adjustments (vs. Overall Personal Baseline):")
    for cat in sorted(
        model.style_categories,
        key=lambda s: model.style_adjustments[s],
        reverse=True,
    ):
        print(f"  • {cat.title():<12}: {model.style_adjustments[cat]:+6.3f} pts")
    print()

    # Aggregate by Beer
    beer_checkin_indices = defaultdict(list)
    for idx, c in enumerate(checkins):
        beer_checkin_indices[c.beer].append(idx)

    beer_summaries = []
    for beer, indices in beer_checkin_indices.items():
        actual_mean = float(np.mean([checkins[i].rating for i in indices]))
        expected_mean = float(np.mean([fitted[i] for i in indices]))
        delta_mean = actual_mean - expected_mean
        z_mean = delta_mean / model.sigma_epsilon
        beer_summaries.append(
            {
                "beer": beer,
                "n": len(indices),
                "actual": actual_mean,
                "expected": expected_mean,
                "delta": delta_mean,
                "z": z_mean,
                "global": beer.global_rating,
                "abv": beer.abv,
                "style": beer.get_style_category(),
            }
        )

    # 1. Top Overperforming Beers
    sorted_over = sorted(beer_summaries, key=lambda b: b["z"], reverse=True)
    print("=" * 110)
    print(
        f" TOP {top_n} OVERPERFORMING BEERS (Exceeded Expectations Given Global Rating, Style, ABV, & Date)"
    )
    print("=" * 110)
    print(
        f"{'#':<3} | {'BEER NAME / BREWERY':<48} | {'ABV':>5} | {'STYLE':<10} | {'GLOBAL':>6} | {'ACTUAL':>6} | {'EXPECTED':>8} | {'DELTA':>6} | {'Z-SCORE':>7}"
    )
    print("-" * 110)
    for rank, b in enumerate(sorted_over[:top_n], 1):
        beer_str = f"{b['beer'].name} ({b['beer'].brewery.name})"
        if len(beer_str) > 48:
            beer_str = beer_str[:45] + "..."
        print(
            f"{rank:<3} | {beer_str:<48} | {b['abv']:>4.1f}% | {b['style']:<10} | {b['global']:>6.2f} | {b['actual']:>6.2f} | {b['expected']:>8.2f} | {b['delta']:>+6.2f} | {b['z']:>+7.2f}σ"
        )
    print()

    # 2. Top Underperforming Beers
    sorted_under = sorted(beer_summaries, key=lambda b: b["z"])
    print("=" * 110)
    print(
        f" TOP {top_n} UNDERPERFORMING BEERS (Fell Short of Expectations Given Global Rating, Style, ABV, & Date)"
    )
    print("=" * 110)
    print(
        f"{'#':<3} | {'BEER NAME / BREWERY':<48} | {'ABV':>5} | {'STYLE':<10} | {'GLOBAL':>6} | {'ACTUAL':>6} | {'EXPECTED':>8} | {'DELTA':>6} | {'Z-SCORE':>7}"
    )
    print("-" * 110)
    for rank, b in enumerate(sorted_under[:top_n], 1):
        beer_str = f"{b['beer'].name} ({b['beer'].brewery.name})"
        if len(beer_str) > 48:
            beer_str = beer_str[:45] + "..."
        print(
            f"{rank:<3} | {beer_str:<48} | {b['abv']:>4.1f}% | {b['style']:<10} | {b['global']:>6.2f} | {b['actual']:>6.2f} | {b['expected']:>8.2f} | {b['delta']:>+6.2f} | {b['z']:>+7.2f}σ"
        )
    print()

    # 3. Brewery Analysis
    brewery_indices = defaultdict(list)
    for idx, c in enumerate(checkins):
        brewery_indices[c.beer.brewery.name].append(idx)

    brewery_summaries = []
    for brew_name, indices in brewery_indices.items():
        if len(indices) < min_brewery_checkins:
            continue
        actual_mean = float(np.mean([checkins[i].rating for i in indices]))
        expected_mean = float(np.mean([fitted[i] for i in indices]))
        global_mean = float(np.mean([checkins[i].beer.global_rating for i in indices]))
        delta_mean = actual_mean - expected_mean
        z_mean = delta_mean / model.sigma_epsilon
        brewery_summaries.append(
            {
                "name": brew_name,
                "n": len(indices),
                "actual": actual_mean,
                "expected": expected_mean,
                "global": global_mean,
                "delta": delta_mean,
                "z": z_mean,
            }
        )

    sorted_brew_over = sorted(brewery_summaries, key=lambda b: b["z"], reverse=True)
    sorted_brew_under = sorted(brewery_summaries, key=lambda b: b["z"])

    print("=" * 110)
    print(f" TOP OVERPERFORMING BREWERIES (Min {min_brewery_checkins} Check-ins)")
    print("=" * 110)
    print(
        f"{'#':<3} | {'BREWERY NAME':<38} | {'N':>4} | {'GLOBAL':>6} | {'ACTUAL':>6} | {'EXPECTED':>8} | {'MEAN DELTA':>10} | {'MEAN Z':>8}"
    )
    print("-" * 110)
    for rank, b in enumerate(sorted_brew_over[:top_n], 1):
        name_str = b["name"][:38]
        print(
            f"{rank:<3} | {name_str:<38} | {b['n']:>4} | {b['global']:>6.2f} | {b['actual']:>6.2f} | {b['expected']:>8.2f} | {b['delta']:>+10.2f} | {b['z']:>+7.2f}σ"
        )
    print()

    print("=" * 110)
    print(f" TOP UNDERPERFORMING BREWERIES (Min {min_brewery_checkins} Check-ins)")
    print("=" * 110)
    print(
        f"{'#':<3} | {'BREWERY NAME':<38} | {'N':>4} | {'GLOBAL':>6} | {'ACTUAL':>6} | {'EXPECTED':>8} | {'MEAN DELTA':>10} | {'MEAN Z':>8}"
    )
    print("-" * 110)
    for rank, b in enumerate(sorted_brew_under[:top_n], 1):
        name_str = b["name"][:38]
        print(
            f"{rank:<3} | {name_str:<38} | {b['n']:>4} | {b['global']:>6.2f} | {b['actual']:>6.2f} | {b['expected']:>8.2f} | {b['delta']:>+10.2f} | {b['z']:>+7.2f}σ"
        )
    print("=" * 110)


if __name__ == "__main__":
    checkins_raw = untappd.load_latest_checkins()
    checkins = [
        c
        for c in checkins_raw
        if c.rating is not None
        and c.beer.global_rating is not None
        and c.beer.global_rating > 0
        and c.beer.abv is not None
        and c.datetime is not None
    ]

    model, fitted, residuals, z_residuals = fit_multivariate_model(checkins)

    print_rankings_and_report(
        checkins,
        model,
        fitted,
        residuals,
        z_residuals,
        top_n=25,
        min_brewery_checkins=5,
    )

    out_plot = (
        Path(__file__).resolve().parent
        / "out"
        / "top_beers_normalized_multivariate.png"
    )
    plot_multivariate_decomposition(
        checkins,
        model,
        z_residuals,
        out_file=out_plot,
    )
