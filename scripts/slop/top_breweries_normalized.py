from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Ensure untappd package is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import untappd
import untappd_utils

# Reconfigure stdout to use UTF-8 on Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


@dataclass
class NormalizationResult:
    normalized_ratings: dict[untappd.Beer, float]
    all_mean: float
    all_std: float
    mode: str
    metadata: dict[str, object]


def compute_magic_rating(
    beer_ratings: Sequence[tuple[untappd.Beer, float]],
    dropoff_ratio: float = 0.8,
    average_score_weight: float = 0.35,
    neutral_rating: float = 3.25,
) -> tuple[float, list[tuple[untappd.Beer, float]]]:
    """
    Computes magic rating score and sorted beer list from (beer, rating) pairs,
    matching the exact mathematical formulation of untappd.magic_rating.
    """
    if not beer_ratings:
        return 0.0, []
    top_ratings = sorted(beer_ratings, key=lambda t: t[1], reverse=True)
    ratings = [r for _, r in top_ratings]
    avg_rating = sum(ratings) / len(ratings)
    geometric_sum = neutral_rating + (1.0 - dropoff_ratio) * sum(
        (r - neutral_rating) * (dropoff_ratio**i) for i, r in enumerate(ratings)
    )
    score = (
        average_score_weight * avg_rating + (1.0 - average_score_weight) * geometric_sum
    )
    return score, top_ratings


def normalize_by_style(
    ratings_by_beer: dict[untappd.Beer, float],
    all_mean: float,
    all_std: float,
) -> NormalizationResult:
    ratings_by_cat = defaultdict(list)
    for beer, r in ratings_by_beer.items():
        ratings_by_cat[beer.get_style_category()].append(r)

    stats_by_cat = {}
    for cat, rs in ratings_by_cat.items():
        cat_mean = float(np.mean(rs))
        cat_std = float(np.std(rs, ddof=1)) if len(rs) > 1 else all_std
        # Guard against zero variance in tiny categories
        stats_by_cat[cat] = (cat_mean, cat_std if cat_std > 1e-4 else all_std)

    normalized = {}
    for beer, r in ratings_by_beer.items():
        cat = beer.get_style_category()
        c_mean, c_std = stats_by_cat.get(cat, (all_mean, all_std))
        normalized[beer] = all_mean + all_std * (r - c_mean) / c_std

    return NormalizationResult(
        normalized_ratings=normalized,
        all_mean=all_mean,
        all_std=all_std,
        mode="style",
        metadata={"stats_by_cat": stats_by_cat},
    )


def normalize_by_abv(
    ratings_by_beer: dict[untappd.Beer, float],
    all_mean: float,
    all_std: float,
    method: str = "regression",
) -> NormalizationResult:
    beers = list(ratings_by_beer.keys())
    raw_ratings = np.array([ratings_by_beer[b] for b in beers], dtype=float)
    abvs = np.array([b.abv if b.abv is not None else 5.0 for b in beers], dtype=float)

    if method == "bucket":
        upper_bounds = np.linspace(0, 14.5, 21)

        def round_abv(val: float) -> float:
            return next((x for x in upper_bounds if x >= val), upper_bounds[-1])

        ratings_by_bucket = defaultdict(list)
        for b, r in ratings_by_beer.items():
            ratings_by_bucket[round_abv(b.abv if b.abv is not None else 5.0)].append(r)

        stats_by_bucket = {
            b_val: (
                float(np.mean(rs)),
                float(np.std(rs, ddof=1)) if len(rs) > 1 else all_std,
            )
            for b_val, rs in ratings_by_bucket.items()
        }

        normalized = {}
        for b, r in ratings_by_beer.items():
            b_val = round_abv(b.abv if b.abv is not None else 5.0)
            b_mean, b_std = stats_by_bucket.get(b_val, (all_mean, all_std))
            b_std_safe = b_std if b_std > 1e-4 else all_std
            normalized[b] = all_mean + all_std * (r - b_mean) / b_std_safe

        return NormalizationResult(
            normalized_ratings=normalized,
            all_mean=all_mean,
            all_std=all_std,
            mode="abv_bucket",
            metadata={"stats_by_bucket": stats_by_bucket},
        )

    # Quadratic polynomial regression: E[rating | ABV] = c0 + c1*ABV + c2*ABV^2
    poly_coeffs = np.polyfit(abvs, raw_ratings, 2)
    poly = np.poly1d(poly_coeffs)
    residuals = raw_ratings - poly(abvs)
    res_std = float(np.std(residuals, ddof=3)) if len(residuals) > 3 else all_std
    res_std_safe = res_std if res_std > 1e-4 else all_std

    normalized = {
        b: all_mean + all_std * (residuals[i] / res_std_safe)
        for i, b in enumerate(beers)
    }

    return NormalizationResult(
        normalized_ratings=normalized,
        all_mean=all_mean,
        all_std=all_std,
        mode="abv",
        metadata={
            "poly_coeffs": poly_coeffs.tolist(),
            "res_std": res_std_safe,
        },
    )


def normalize_by_both(
    ratings_by_beer: dict[untappd.Beer, float],
    all_mean: float,
    all_std: float,
) -> NormalizationResult:
    beers = list(ratings_by_beer.keys())
    raw_ratings = np.array([ratings_by_beer[b] for b in beers], dtype=float)
    abvs = np.array([b.abv if b.abv is not None else 5.0 for b in beers], dtype=float)
    categories = sorted(list(set(b.get_style_category() for b in beers)))
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Design matrix: [One-hot styles, (ABV - mean_abv), (ABV - mean_abv)^2]
    mean_abv = float(np.mean(abvs))
    x_mat = np.zeros((len(beers), len(categories) + 2), dtype=float)
    for i, b in enumerate(beers):
        x_mat[i, cat_to_idx[b.get_style_category()]] = 1.0
        centered_abv = abvs[i] - mean_abv
        x_mat[i, -2] = centered_abv
        x_mat[i, -1] = centered_abv**2

    coefs, _, _, _ = np.linalg.lstsq(x_mat, raw_ratings, rcond=None)
    fitted = x_mat @ coefs
    residuals = raw_ratings - fitted
    df_resid = max(1, len(beers) - len(coefs))
    res_std = float(np.sqrt(np.sum(residuals**2) / df_resid))
    res_std_safe = res_std if res_std > 1e-4 else all_std

    ss_tot = np.sum((raw_ratings - np.mean(raw_ratings)) ** 2)
    ss_res = np.sum(residuals**2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    normalized = {
        b: all_mean + all_std * (residuals[i] / res_std_safe)
        for i, b in enumerate(beers)
    }

    style_coefs = {cat: float(coefs[cat_to_idx[cat]]) for cat in categories}

    return NormalizationResult(
        normalized_ratings=normalized,
        all_mean=all_mean,
        all_std=all_std,
        mode="both",
        metadata={
            "mean_abv": mean_abv,
            "style_coefs": style_coefs,
            "abv_linear_coef": float(coefs[-2]),
            "abv_quad_coef": float(coefs[-1]),
            "res_std": res_std_safe,
            "r_squared": r_squared,
        },
    )


def get_style_ratios(checkins: Sequence[untappd.Checkin]) -> list[tuple[str, float]]:
    style_counts = Counter(c.beer.get_style_category() for c in checkins)
    total = len(checkins)
    if total == 0:
        return []
    return [
        (style, count / total)
        for style, count in sorted(
            style_counts.items(), key=lambda t: t[1], reverse=True
        )
    ]


def rank_breweries(
    checkins: Sequence[untappd.Checkin],
    norm_result: NormalizationResult,
    ratings_by_beer: dict[untappd.Beer, float],
    specific_countries: Sequence[str] = (),
    specific_styles: Sequence[str] = (),
    dropoff_ratio: float = 0.8,
    average_score_weight: float = 0.35,
    neutral_rating: float = 3.25,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    # Group checkins by brewery
    brewery_checkins = defaultdict(list)
    for c in checkins:
        if (
            specific_styles
            and c.beer.get_style_category().lower() not in specific_styles
        ):
            continue
        brewery = c.beer.brewery
        if specific_countries and not any(
            c_str in brewery.country.lower() for c_str in specific_countries
        ):
            continue
        brewery_checkins[brewery].append(c)

    scored_breweries = []
    for brewery, b_checkins in brewery_checkins.items():
        distinct_beers = set(c.beer for c in b_checkins)

        # Normalized magic rating
        norm_beers = [
            (b, norm_result.normalized_ratings.get(b, norm_result.all_mean))
            for b in distinct_beers
        ]
        norm_score, norm_top_beers = compute_magic_rating(
            norm_beers,
            dropoff_ratio=dropoff_ratio,
            average_score_weight=average_score_weight,
            neutral_rating=neutral_rating,
        )

        # Raw magic rating
        raw_beers = [
            (b, ratings_by_beer.get(b, norm_result.all_mean)) for b in distinct_beers
        ]
        raw_score, raw_top_beers = compute_magic_rating(
            raw_beers,
            dropoff_ratio=dropoff_ratio,
            average_score_weight=average_score_weight,
            neutral_rating=neutral_rating,
        )

        scored_breweries.append(
            {
                "brewery": brewery,
                "checkins": b_checkins,
                "n_checkins": len(b_checkins),
                "n_beers": len(distinct_beers),
                "norm_score": norm_score,
                "norm_top_beers": norm_top_beers,
                "raw_score": raw_score,
                "raw_top_beers": raw_top_beers,
            }
        )

    # Calculate ranks to track climbers and decliners
    raw_sorted = sorted(scored_breweries, key=lambda d: d["raw_score"], reverse=True)
    for r, item in enumerate(raw_sorted, 1):
        item["raw_rank"] = r

    norm_sorted = sorted(scored_breweries, key=lambda d: d["norm_score"], reverse=True)
    for r, item in enumerate(norm_sorted, 1):
        item["norm_rank"] = r
        item["rank_delta"] = item["raw_rank"] - r
        item["score_delta"] = item["norm_score"] - item["raw_score"]

    return norm_sorted, raw_sorted


def format_ranking_report(
    norm_sorted: list[dict[str, object]],
    norm_result: NormalizationResult,
    ratings_by_beer: dict[untappd.Beer, float],
    all_checkins: Sequence[untappd.Checkin],
    top_n_breweries: int = 20,
    top_n_beers: int = 5,
    min_checkins_for_movers: int = 5,
    specific_styles: Sequence[str] = (),
    specific_countries: Sequence[str] = (),
) -> str:
    lines = []
    lines.append("=" * 88)
    lines.append(f"TOP BREWERIES - NORMALIZED BY {norm_result.mode.upper()}")
    lines.append("=" * 88)

    if specific_countries:
        lines.append(f"Filtered Countries: {', '.join(specific_countries)}")
    if specific_styles:
        lines.append(f"Filtered Styles: {', '.join(specific_styles)}")

    if norm_result.mode == "both":
        meta = norm_result.metadata
        lines.append(
            f"Multivariate Model Fit: R² = {meta['r_squared']:.3f}, "
            f"Residual Std = {meta['res_std']:.3f}"
        )
        lines.append(
            f"ABV Partial Effect: Linear = {meta['abv_linear_coef']:+.4f}/%, "
            f"Quadratic = {meta['abv_quad_coef']:+.5f}"
        )
        lines.append(
            "Style Baselines (vs mean): "
            + ", ".join(
                f"{k.capitalize()} {v - norm_result.all_mean:+.2f}"
                for k, v in meta["style_coefs"].items()
            )
        )
    elif norm_result.mode == "style":
        lines.append(
            "Style Category Baselines: "
            + ", ".join(
                f"{k.capitalize()} μ={m:.2f} (σ={s:.2f})"
                for k, (m, s) in norm_result.metadata["stats_by_cat"].items()
            )
        )
    elif norm_result.mode == "abv":
        lines.append(
            f"ABV Regression Fit: Residual Std = {norm_result.metadata['res_std']:.3f}"
        )
    lines.append("")

    for rank, item in enumerate(norm_sorted[:top_n_breweries], 1):
        brewery = item["brewery"]
        rank_delta_str = (
            f"rank {item['rank_delta']:+d}"
            if item["rank_delta"] != 0
            else "rank unchanged"
        )
        lines.append(
            f"#{rank:<3} {brewery} (magic rating: {item['norm_score']:.2f}, "
            f"raw: {item['raw_score']:.2f} [{item['score_delta']:+.2f}, {rank_delta_str}] "
            f"over {item['n_checkins']} checkins, {item['n_beers']} unique beers)"
        )

        lines.append(
            "Styles across all checkins: "
            + ", ".join(
                f"{style.capitalize()} {ratio:.1%}"
                for style, ratio in get_style_ratios(item["checkins"])
            )
        )

        if top_n_beers > 0:
            for beer, norm_r in item["norm_top_beers"][:top_n_beers]:
                raw_r = ratings_by_beer.get(beer, norm_r)
                beer_delta = norm_r - raw_r
                lines.append(
                    f"{norm_r:.2f} [raw {raw_r:.2f}, {beer_delta:+.2f}]  {beer}"
                )
            lines.append("")

    # Global benchmark scores
    all_beers = [
        (b, norm_result.normalized_ratings.get(b, norm_result.all_mean))
        for b in ratings_by_beer
    ]
    all_norm_score, _ = compute_magic_rating(all_beers)
    all_raw_beers = list(ratings_by_beer.items())
    all_raw_score, _ = compute_magic_rating(all_raw_beers)
    lines.append(
        f"(magic rating across all beers: normalized {all_norm_score:.2f}, "
        f"raw {all_raw_score:.2f})"
    )
    lines.append("")

    # Biggest Climbers and Decliners
    eligible = [
        item for item in norm_sorted if item["n_checkins"] >= min_checkins_for_movers
    ]

    lines.append("=" * 88)
    lines.append(
        f"BIGGEST CLIMBERS (Gaining most rank after normalization, "
        f"min {min_checkins_for_movers} checkins)"
    )
    lines.append("=" * 88)
    climbers = sorted(eligible, key=lambda d: d["rank_delta"], reverse=True)
    for rank, item in enumerate(climbers[:10], 1):
        brewery = item["brewery"]
        lines.append(
            f"{rank:<2}. {brewery.name:<34} | +{item['rank_delta']:<4} ranks "
            f"(#{item['raw_rank']} -> #{item['norm_rank']}) | "
            f"norm: {item['norm_score']:.2f} (raw: {item['raw_score']:.2f}) | "
            f"{item['n_checkins']:>3} checkins"
        )
    lines.append("")

    lines.append("=" * 88)
    lines.append(
        f"BIGGEST DECLINERS (Dropping most rank after normalization, "
        f"min {min_checkins_for_movers} checkins)"
    )
    lines.append("=" * 88)
    decliners = sorted(eligible, key=lambda d: d["rank_delta"])
    for rank, item in enumerate(decliners[:10], 1):
        brewery = item["brewery"]
        lines.append(
            f"{rank:<2}. {brewery.name:<34} | {item['rank_delta']:<5} ranks "
            f"(#{item['raw_rank']} -> #{item['norm_rank']}) | "
            f"norm: {item['norm_score']:.2f} (raw: {item['raw_score']:.2f}) | "
            f"{item['n_checkins']:>3} checkins"
        )
    lines.append("")

    return "\n".join(lines)


def plot_normalization_diagnostics(
    norm_sorted: list[dict[str, object]],
    norm_result: NormalizationResult,
    out_file: Path,
    min_checkins: int = 5,
):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#f9f9f9")

    eligible = [item for item in norm_sorted if item["n_checkins"] >= min_checkins]
    raw_scores = [item["raw_score"] for item in eligible]
    norm_scores = [item["norm_score"] for item in eligible]
    deltas = [item["rank_delta"] for item in eligible]

    # Panel 1: Scatter of Raw vs Normalized Magic Rating
    ax1 = axes[0]
    ax1.set_facecolor("white")
    scatter = ax1.scatter(
        raw_scores,
        norm_scores,
        c=deltas,
        cmap="coolwarm",
        alpha=0.75,
        edgecolors="none",
        s=36,
    )
    fig.colorbar(scatter, ax=ax1, label="Rank Shift (+ = Climbed, - = Dropped)")
    min_val = min(min(raw_scores), min(norm_scores)) - 0.1
    max_val = max(max(raw_scores), max(norm_scores)) + 0.1
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        label="y = x (No change)",
    )

    # Label top 5 normalized breweries
    for item in norm_sorted[:5]:
        ax1.annotate(
            item["brewery"].name,
            (item["raw_score"], item["norm_score"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    ax1.set_title(f"Brewery Magic Rating: Raw vs {norm_result.mode.upper()}-Normalized")
    ax1.set_xlabel("Raw Magic Rating")
    ax1.set_ylabel("Normalized Magic Rating")
    ax1.legend(loc="upper left")

    # Panel 2: Top movers bar chart
    ax2 = axes[1]
    ax2.set_facecolor("white")
    climbers = sorted(eligible, key=lambda d: d["rank_delta"], reverse=True)[:8]
    decliners = sorted(eligible, key=lambda d: d["rank_delta"])[:8]
    movers = climbers + decliners[::-1]

    names = [m["brewery"].name[:24] for m in movers]
    rank_shifts = [m["rank_delta"] for m in movers]
    colors = ["#2ca02c" if s > 0 else "#d62728" for s in rank_shifts]

    ax2.barh(range(len(names)), rank_shifts, color=colors, alpha=0.85)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xlabel("Rank Shift (Ranks Gained / Lost)")
    ax2.set_title("Top Climbers & Decliners after Normalization")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Diagnostic plot saved to: {out_file}")


def run_ranking(
    normalize_mode: str = "both",
    top_breweries: int = 20,
    top_beers: int = 5,
    min_checkins_movers: int = 5,
    specific_styles: Sequence[str] = (),
    specific_countries: Sequence[str] = (),
    save_plot: bool = False,
    out_dir: Optional[Path] = None,
) -> str:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkins = untappd.load_latest_checkins()
    ratings_by_beer = untappd.average_rating_by_beer(checkins)
    all_ratings = list(ratings_by_beer.values())
    all_mean = float(np.mean(all_ratings))
    all_std = float(np.std(all_ratings, ddof=1))

    if normalize_mode == "none":
        norm_result = NormalizationResult(
            normalized_ratings=ratings_by_beer,
            all_mean=all_mean,
            all_std=all_std,
            mode="none",
            metadata={},
        )
    elif normalize_mode == "style":
        norm_result = normalize_by_style(ratings_by_beer, all_mean, all_std)
    elif normalize_mode == "abv":
        norm_result = normalize_by_abv(ratings_by_beer, all_mean, all_std)
    elif normalize_mode == "both":
        norm_result = normalize_by_both(ratings_by_beer, all_mean, all_std)
    else:
        raise ValueError(f"Unknown normalize mode: {normalize_mode}")

    norm_sorted, _ = rank_breweries(
        checkins=checkins,
        norm_result=norm_result,
        ratings_by_beer=ratings_by_beer,
        specific_countries=specific_countries,
        specific_styles=specific_styles,
    )

    report = format_ranking_report(
        norm_sorted=norm_sorted,
        norm_result=norm_result,
        ratings_by_beer=ratings_by_beer,
        all_checkins=checkins,
        top_n_breweries=top_breweries,
        top_n_beers=top_beers,
        min_checkins_for_movers=min_checkins_movers,
        specific_styles=specific_styles,
        specific_countries=specific_countries,
    )

    out_text_file = out_dir / f"top_breweries_normalized_{normalize_mode}.txt"
    out_text_file.write_text(report, encoding="UTF-8")
    print(report)
    print(f"Results written to: {out_text_file}")

    if save_plot:
        out_plot_file = out_dir / f"top_breweries_normalized_{normalize_mode}.png"
        plot_normalization_diagnostics(norm_sorted, norm_result, out_plot_file)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Score and rank breweries with magic rating normalized by ABV and/or style."
    )
    parser.add_argument(
        "--normalize",
        "-n",
        choices=["both", "style", "abv", "none", "all"],
        default="both",
        help="Normalization dimension: both (ABV & style), style, abv, none, or all (default: both).",
    )
    parser.add_argument(
        "--top-breweries",
        type=int,
        default=20,
        help="Number of top breweries to display (default: 20).",
    )
    parser.add_argument(
        "--top-beers",
        type=int,
        default=5,
        help="Number of top beers per brewery to display (default: 5).",
    )
    parser.add_argument(
        "--min-checkins",
        type=int,
        default=5,
        help="Minimum checkins threshold for climbers/decliners tables (default: 5).",
    )
    parser.add_argument(
        "--countries",
        nargs="*",
        default=[],
        help="Filter by country names (lowercase substring matching).",
    )
    parser.add_argument(
        "--styles",
        nargs="*",
        default=[],
        help="Filter by style categories (e.g. stout, ipa, sour, lager).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save diagnostic comparison plots to out directory.",
    )

    args = parser.parse_args()

    modes = (
        ["both", "style", "abv", "none"]
        if args.normalize == "all"
        else [args.normalize]
    )

    for m in modes:
        run_ranking(
            normalize_mode=m,
            top_breweries=args.top_breweries,
            top_beers=args.top_beers,
            min_checkins_movers=args.min_checkins,
            specific_styles=[s.lower() for s in args.styles],
            specific_countries=[c.lower() for c in args.countries],
            save_plot=args.plot,
        )


if __name__ == "__main__":
    main()
