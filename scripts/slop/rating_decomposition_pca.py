import datetime
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import untappd
import untappd_categorise

# Ensure headless plotting
plt.switch_backend("Agg")


def natural_cubic_spline_basis(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """
    Constructs a Natural Cubic Spline (NCS) basis matrix for 1D input x.
    Linearly extrapolated past boundary knots.
    """
    knots = np.unique(np.asarray(knots, dtype=float))
    x = np.asarray(x, dtype=float)
    if len(knots) < 3:
        return x.reshape(-1, 1)

    k_K = knots[-1]
    k_Km1 = knots[-2]
    basis = [x]

    def d_k(k):
        denom = k_K - k
        if abs(denom) < 1e-9:
            return np.zeros_like(x)
        term1 = np.maximum(0.0, x - k) ** 3
        term2 = np.maximum(0.0, x - k_K) ** 3
        return (term1 - term2) / denom

    d_Km1 = d_k(k_Km1)
    for k in knots[:-2]:
        basis.append(d_k(k) - d_Km1)

    return np.column_stack(basis)


STYLE_KEYWORDS = [
    "imperial",
    "double",
    "triple",
    "pastry",
    "barrel-aged",
    "hazy",
    "new england",
    "fruited",
    "smoothie",
    "sour",
    "wild",
    "brett",
    "coffee",
    "chocolate",
    "vanilla",
    "milk",
    "session",
    "dry",
    "west coast",
    "black",
    "smoked",
]


def extract_features(checkins: list[untappd.Checkin]):
    """
    Extracts structured intrinsic and extrinsic feature sets for all check-ins.
    """
    session_tracker = untappd_categorise.SessionTracker(
        cap=10, max_gap=datetime.timedelta(hours=3)
    )
    nth_tracker = untappd_categorise.BeerNthTimeTracker(max_n=5)

    n = len(checkins)
    ratings = np.array([float(c.rating) for c in checkins], dtype=float)

    # 1. Extrinsic temporal & session
    session_nums = []
    beer_nths = []
    hours = []
    days_since_start = []
    min_dt = min(c.datetime for c in checkins)
    friend_counts = []
    venues = []
    serving_types = []

    for c in checkins:
        s_raw = session_tracker.session_n(c)
        try:
            s_val = int(str(s_raw).replace("+", ""))
        except ValueError:
            s_val = 1
        session_nums.append(s_val)

        nth_raw = nth_tracker.beer_n(c)
        try:
            nth_val = int(str(nth_raw).replace("+", ""))
        except ValueError:
            nth_val = 1
        beer_nths.append(nth_val)

        hours.append(c.datetime.hour + c.datetime.minute / 60.0)
        days_since_start.append((c.datetime - min_dt).total_seconds() / 86400.0)

        n_friends = (
            len(c.tagged_friends.split(","))
            if c.tagged_friends and str(c.tagged_friends).strip()
            else 0
        )
        friend_counts.append(n_friends)

        v_name = c.venue.name.strip() if (c.venue and c.venue.name) else "Home / None"
        venues.append(v_name)

        srv = c.serving_type.strip() if c.serving_type else "Other / Unknown"
        serving_types.append(srv)

    session_nums = np.array(session_nums, dtype=float)
    beer_nths = np.array(beer_nths, dtype=float)
    hours = np.array(hours, dtype=float)
    days_since_start = np.array(days_since_start, dtype=float)
    friend_counts = np.array(friend_counts, dtype=float)

    # 2. Intrinsic attributes
    abvs = np.array(
        [
            (
                float(c.beer.abv)
                if (c.beer and c.beer.abv is not None)
                else np.nanmedian(
                    [x.beer.abv for x in checkins if x.beer and x.beer.abv is not None]
                )
            )
            for c in checkins
        ],
        dtype=float,
    )

    superstyles = [
        (
            c.beer.get_style_category()
            if (c.beer and hasattr(c.beer, "get_style_category"))
            else "other"
        )
        for c in checkins
    ]
    substyles = [
        c.beer.type.strip() if (c.beer and c.beer.type) else "Unknown" for c in checkins
    ]
    breweries = [
        (
            c.beer.brewery.name.strip()
            if (c.beer and c.beer.brewery and c.beer.brewery.name)
            else "Unknown Brewery"
        )
        for c in checkins
    ]
    brewery_countries = [
        (
            c.beer.brewery.country.strip()
            if (c.beer and c.beer.brewery and c.beer.brewery.country)
            else "Unknown"
        )
        for c in checkins
    ]
    beer_names = [
        c.beer.name.strip() if (c.beer and c.beer.name) else "Unknown Beer"
        for c in checkins
    ]

    # Keyword flags
    keyword_matrix = np.zeros((n, len(STYLE_KEYWORDS)), dtype=float)
    for i, sub in enumerate(substyles):
        sub_lower = sub.lower()
        for j, kw in enumerate(STYLE_KEYWORDS):
            if kw in sub_lower:
                keyword_matrix[i, j] = 1.0

    return {
        "n": n,
        "ratings": ratings,
        "abvs": abvs,
        "superstyles": superstyles,
        "substyles": substyles,
        "breweries": breweries,
        "brewery_countries": brewery_countries,
        "beer_names": beer_names,
        "keyword_matrix": keyword_matrix,
        "session_nums": session_nums,
        "beer_nths": beer_nths,
        "hours": hours,
        "days_since_start": days_since_start,
        "friend_counts": friend_counts,
        "venues": venues,
        "serving_types": serving_types,
        "min_dt": min_dt,
    }


def fit_decomposition(features: dict, l2_penalty: float = 10.0):
    """
    Fits the additive linear model using Ridge regression:
    rating = mu + Delta_extrinsic + Delta_intrinsic + epsilon
    """
    n = features["n"]
    y = features["ratings"]
    mu = float(np.mean(y))
    y_centered = y - mu

    # Build Extrinsic Design Matrix X_ext
    # 1. Session drink spline/linear
    session_x = (np.clip(features["session_nums"], 1, 10) - 1.0).reshape(-1, 1)

    # 2. Novelty / repeat ordinal (1st taste vs repeated pints)
    repeat_x = (np.clip(features["beer_nths"], 1, 5) - 1.0).reshape(-1, 1)

    # 3. Diurnal cycle: cyclical 24h sine/cosine
    hour_rad = features["hours"] * (2 * np.pi / 24.0)
    hour_sin = np.sin(hour_rad).reshape(-1, 1)
    hour_cos = np.cos(hour_rad).reshape(-1, 1)

    # 4. Multi-year timeline trend (natural spline with 5 knots)
    t_knots = np.percentile(features["days_since_start"], [5, 25, 50, 75, 95])
    t_basis = natural_cubic_spline_basis(features["days_since_start"], t_knots)
    t_basis -= np.mean(t_basis, axis=0)

    # 5. Tagged friends (indicator of presence + count)
    friends_x = np.column_stack(
        [
            (features["friend_counts"] > 0).astype(float),
            np.clip(features["friend_counts"], 0, 5),
        ]
    )

    # 6. Serving type one-hot (only categories with >= 20 occurrences)
    srv_counts = pd.Series(features["serving_types"]).value_counts()
    srv_levels = [
        s for s in srv_counts.index if srv_counts[s] >= 20 and s != "Other / Unknown"
    ]
    srv_matrix = np.zeros((n, len(srv_levels)), dtype=float)
    for j, lvl in enumerate(srv_levels):
        srv_matrix[:, j] = (np.array(features["serving_types"]) == lvl).astype(float)
    if srv_matrix.shape[1] > 0:
        srv_matrix -= np.mean(srv_matrix, axis=0)

    # 7. Venue effects (shrinkage on all venues with >= 5 check-ins, reference 'Home / None')
    venue_counts = pd.Series(features["venues"]).value_counts()
    venue_levels = [
        v for v in venue_counts.index if venue_counts[v] >= 5 and v != "Home / None"
    ]
    venue_matrix = np.zeros((n, len(venue_levels)), dtype=float)
    for j, lvl in enumerate(venue_levels):
        venue_matrix[:, j] = (np.array(features["venues"]) == lvl).astype(float)
    venue_matrix -= np.mean(venue_matrix, axis=0)

    X_ext = np.column_stack(
        [
            session_x - np.mean(session_x),
            repeat_x - np.mean(repeat_x),
            hour_sin - np.mean(hour_sin),
            hour_cos - np.mean(hour_cos),
            t_basis,
            friends_x - np.mean(friends_x, axis=0),
            *([srv_matrix] if srv_matrix.shape[1] > 0 else []),
            venue_matrix,
        ]
    )

    # Build Intrinsic Design Matrix X_int
    # 1. ABV non-linear natural cubic spline
    abv_knots = np.percentile(features["abvs"], [5, 25, 50, 75, 95])
    abv_basis = natural_cubic_spline_basis(features["abvs"], abv_knots)
    abv_basis -= np.mean(abv_basis, axis=0)

    # 2. Superstyle one-hot
    superstyle_levels = sorted(list(set(features["superstyles"])))
    superstyle_matrix = np.zeros((n, len(superstyle_levels)), dtype=float)
    for j, lvl in enumerate(superstyle_levels):
        superstyle_matrix[:, j] = (np.array(features["superstyles"]) == lvl).astype(
            float
        )
    superstyle_matrix -= np.mean(superstyle_matrix, axis=0)

    # 3. Substyle keywords matrix
    kw_matrix = features["keyword_matrix"] - np.mean(features["keyword_matrix"], axis=0)

    # 4. Top brewery effects (shrinkage on breweries with >= 10 checkins)
    bw_counts = pd.Series(features["breweries"]).value_counts()
    bw_levels = [b for b in bw_counts.index if bw_counts[b] >= 10]
    bw_matrix = np.zeros((n, len(bw_levels)), dtype=float)
    for j, lvl in enumerate(bw_levels):
        bw_matrix[:, j] = (np.array(features["breweries"]) == lvl).astype(float)
    bw_matrix -= np.mean(bw_matrix, axis=0)

    # 5. Brewery country effects (for countries with >= 50 checkins)
    country_counts = pd.Series(features["brewery_countries"]).value_counts()
    country_levels = [
        c for c in country_counts.index if country_counts[c] >= 50 and c != "England"
    ]
    country_matrix = np.zeros((n, len(country_levels)), dtype=float)
    for j, lvl in enumerate(country_levels):
        country_matrix[:, j] = (np.array(features["brewery_countries"]) == lvl).astype(
            float
        )
    country_matrix -= np.mean(country_matrix, axis=0)

    X_int = np.column_stack(
        [
            abv_basis,
            superstyle_matrix,
            kw_matrix,
            country_matrix,
            bw_matrix,
        ]
    )

    # Full design matrix X = [X_ext, X_int]
    p_ext = X_ext.shape[1]
    p_int = X_int.shape[1]
    X_full = np.column_stack([X_ext, X_int])

    # Ridge Regression: beta = (X^T X + lambda I)^(-1) X^T y
    reg_diag = np.ones(X_full.shape[1]) * l2_penalty
    # Apply higher shrinkage penalty to high-cardinality venues and breweries
    venue_start_idx = p_ext - len(venue_levels)
    reg_diag[venue_start_idx:p_ext] = l2_penalty * 2.5
    bw_start_idx = (
        p_ext
        + abv_basis.shape[1]
        + superstyle_matrix.shape[1]
        + kw_matrix.shape[1]
        + country_matrix.shape[1]
    )
    reg_diag[bw_start_idx:] = l2_penalty * 2.0

    XtX = X_full.T @ X_full
    A = XtX + np.diag(reg_diag)
    Xty = X_full.T @ y_centered
    beta = np.linalg.solve(A, Xty)

    beta_ext = beta[:p_ext]
    beta_int = beta[p_ext:]

    delta_extrinsic = X_ext @ beta_ext
    delta_intrinsic = X_int @ beta_int

    # Break out individual extrinsic components for inspection
    # Session: col 0
    delta_session = (X_ext[:, 0]) * beta_ext[0]
    # Novelty: col 1
    delta_novelty = (X_ext[:, 1]) * beta_ext[1]
    # Diurnal: cols 2, 3
    delta_diurnal = X_ext[:, 2:4] @ beta_ext[2:4]
    # Timeline drift: t_basis
    t_dim = t_basis.shape[1]
    delta_timeline = X_ext[:, 4 : 4 + t_dim] @ beta_ext[4 : 4 + t_dim]
    # Friends: 4+t_dim : 4+t_dim+2
    f_start = 4 + t_dim
    delta_friends = X_ext[:, f_start : f_start + 2] @ beta_ext[f_start : f_start + 2]
    # Serving:
    srv_start = f_start + 2
    srv_dim = srv_matrix.shape[1]
    delta_serving = (
        X_ext[:, srv_start : srv_start + srv_dim]
        @ beta_ext[srv_start : srv_start + srv_dim]
        if srv_dim > 0
        else np.zeros(n)
    )
    # Venue:
    delta_venue = X_ext[:, venue_start_idx:p_ext] @ beta_ext[venue_start_idx:p_ext]

    # Break out individual intrinsic components:
    # ABV:
    abv_dim = abv_basis.shape[1]
    delta_abv = X_int[:, :abv_dim] @ beta_int[:abv_dim]
    # Superstyle:
    style_dim = superstyle_matrix.shape[1]
    delta_style = (
        X_int[:, abv_dim : abv_dim + style_dim]
        @ beta_int[abv_dim : abv_dim + style_dim]
    )
    # Keywords:
    kw_dim = kw_matrix.shape[1]
    delta_keywords = (
        X_int[:, abv_dim + style_dim : abv_dim + style_dim + kw_dim]
        @ beta_int[abv_dim + style_dim : abv_dim + style_dim + kw_dim]
    )
    # Brewery country:
    country_dim = country_matrix.shape[1]
    delta_country = (
        X_int[
            :, abv_dim + style_dim + kw_dim : abv_dim + style_dim + kw_dim + country_dim
        ]
        @ beta_int[
            abv_dim + style_dim + kw_dim : abv_dim + style_dim + kw_dim + country_dim
        ]
    )
    # Brewery:
    delta_brewery = X_int[:, bw_start_idx - p_ext :] @ beta_int[bw_start_idx - p_ext :]

    # Residual epsilon:
    epsilon = y - (mu + delta_extrinsic + delta_intrinsic)

    # Neutralized Ratings:
    # rating_extrinsic_neutral: rating purged of all extrinsic environmental factors
    rating_extrinsic_neutral = y - delta_extrinsic

    # rating_style_normalized: also subtracts the style/ABV macro premium (pure outlier / quality within style)
    rating_style_normalized = rating_extrinsic_neutral - mu - (delta_style + delta_abv)

    return {
        "mu": mu,
        "y": y,
        "delta_extrinsic": delta_extrinsic,
        "delta_intrinsic": delta_intrinsic,
        "delta_session": delta_session,
        "delta_novelty": delta_novelty,
        "delta_diurnal": delta_diurnal,
        "delta_timeline": delta_timeline,
        "delta_friends": delta_friends,
        "delta_serving": delta_serving,
        "delta_venue": delta_venue,
        "delta_abv": delta_abv,
        "delta_style": delta_style,
        "delta_keywords": delta_keywords,
        "delta_country": delta_country,
        "delta_brewery": delta_brewery,
        "epsilon": epsilon,
        "rating_extrinsic_neutral": rating_extrinsic_neutral,
        "rating_style_normalized": rating_style_normalized,
        "venue_levels": venue_levels,
        "venue_betas": beta_ext[venue_start_idx:p_ext],
        "bw_levels": bw_levels,
        "bw_betas": beta_int[bw_start_idx - p_ext :],
        "X_int": X_int,
        "X_ext": X_ext,
        "abv_knots": abv_knots,
        "t_knots": t_knots,
        "r_squared": 1.0 - np.var(epsilon) / np.var(y),
    }


def compute_pca(X: np.ndarray, n_components: int = 4):
    """
    Computes PCA on a standardized design matrix X using SVD.
    """
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    scores = U[:, :n_components] * S[:n_components]
    loadings = Vt[:n_components, :]
    variance_explained = (S[:n_components] ** 2) / (np.sum(S**2) + 1e-9)
    return scores, loadings, variance_explained


def get_analysis(
    checkins: Optional[list[untappd.Checkin]] = None,
    l2_penalty: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Full in-memory pipeline. Returns (df_decomposed, model_results).
    Runs in ~1-2 seconds for ~13,500 check-ins.
    """
    if checkins is None:
        checkins = untappd.load_latest_checkins()

    features = extract_features(checkins)
    res = fit_decomposition(features, l2_penalty=l2_penalty)

    # Compute Intrinsic and Extrinsic PCA
    int_scores, int_loadings, int_var = compute_pca(res["X_int"], n_components=3)
    ext_scores, ext_loadings, ext_var = compute_pca(res["X_ext"], n_components=3)

    res["int_pca"] = {
        "scores": int_scores,
        "loadings": int_loadings,
        "var_explained": int_var,
    }
    res["ext_pca"] = {
        "scores": ext_scores,
        "loadings": ext_loadings,
        "var_explained": ext_var,
    }

    df = pd.DataFrame(
        {
            "beer_name": features["beer_names"],
            "brewery": features["breweries"],
            "brewery_country": features["brewery_countries"],
            "superstyle": features["superstyles"],
            "substyle": features["substyles"],
            "abv": features["abvs"],
            "venue": features["venues"],
            "serving_type": features["serving_types"],
            "session_num": features["session_nums"],
            "beer_nth": features["beer_nths"],
            "hour": features["hours"],
            "friends_tagged": features["friend_counts"],
            "rating_raw": res["y"],
            "rating_extrinsic_neutral": np.round(res["rating_extrinsic_neutral"], 4),
            "rating_style_normalized": np.round(res["rating_style_normalized"], 4),
            "delta_extrinsic": np.round(res["delta_extrinsic"], 4),
            "delta_intrinsic": np.round(res["delta_intrinsic"], 4),
            "delta_venue": np.round(res["delta_venue"], 4),
            "delta_session": np.round(res["delta_session"], 4),
            "delta_novelty": np.round(res["delta_novelty"], 4),
            "delta_diurnal": np.round(res["delta_diurnal"], 4),
            "delta_timeline": np.round(res["delta_timeline"], 4),
            "delta_friends": np.round(res["delta_friends"], 4),
            "delta_serving": np.round(res["delta_serving"], 4),
            "delta_abv": np.round(res["delta_abv"], 4),
            "delta_style": np.round(res["delta_style"], 4),
            "delta_keywords": np.round(res["delta_keywords"], 4),
            "delta_country": np.round(res["delta_country"], 4),
            "delta_brewery": np.round(res["delta_brewery"], 4),
            "epsilon": np.round(res["epsilon"], 4),
            "int_PC1": np.round(int_scores[:, 0], 4),
            "int_PC2": np.round(int_scores[:, 1], 4),
            "int_PC3": np.round(int_scores[:, 2], 4),
            "ext_PC1": np.round(ext_scores[:, 0], 4),
            "ext_PC2": np.round(ext_scores[:, 1], 4),
        }
    )

    return df, res


def generate_plots(df: pd.DataFrame, res: dict, out_dir: Optional[Path] = None):
    """
    Generates standalone diagnostic plots answering separate analytical questions.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # -------------------------------------------------------------
    # Plot 1: Extrinsic Impact Curves (Session, Diurnal, Timeline, Novelty)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Session progression effect
    sns.lineplot(
        data=df,
        x="session_num",
        y="delta_session",
        ax=axes[0, 0],
        color="#1f77b4",
        marker="o",
        linewidth=2.5,
    )
    axes[0, 0].set_title("Session Progression Impact (Drinks Consumed)")
    axes[0, 0].set_xlabel("Drink # in Session")
    axes[0, 0].set_ylabel(r"$\Delta$ Rating Contribution")
    axes[0, 0].axhline(0, color="gray", linestyle="--", alpha=0.7)

    # (b) Time of Day / Diurnal cycle
    df_hour_bin = df.copy()
    df_hour_bin["hour_int"] = np.floor(df_hour_bin["hour"]).astype(int)
    hour_means = df_hour_bin.groupby("hour_int")["delta_diurnal"].mean().reset_index()
    sns.lineplot(
        data=hour_means,
        x="hour_int",
        y="delta_diurnal",
        ax=axes[0, 1],
        color="#ff7f0e",
        marker="s",
        linewidth=2.5,
    )
    axes[0, 1].set_title("Diurnal Cycle Impact (Hour of Day)")
    axes[0, 1].set_xlabel("Hour of Day (24h)")
    axes[0, 1].set_ylabel(r"$\Delta$ Rating Contribution")
    axes[0, 1].axhline(0, color="gray", linestyle="--", alpha=0.7)

    # (c) Timeline drift (over the entire history)
    sns.lineplot(
        data=df,
        x=df.index,
        y="delta_timeline",
        ax=axes[1, 0],
        color="#2ca02c",
        linewidth=2,
    )
    axes[1, 0].set_title("Longitudinal Palate Drift Over Time")
    axes[1, 0].set_xlabel("Chronological Checkin Index")
    axes[1, 0].set_ylabel(r"$\Delta$ Rating Contribution")
    axes[1, 0].axhline(0, color="gray", linestyle="--", alpha=0.7)

    # (d) Novelty decay vs Repeat check-ins
    sns.barplot(
        data=df,
        x="beer_nth",
        y="delta_novelty",
        ax=axes[1, 1],
        palette="crest",
    )
    axes[1, 1].set_title("Novelty / Repeat Checkin Impact")
    axes[1, 1].set_xlabel("Ordinal Checkin for Same Beer (1st vs repeat)")
    axes[1, 1].set_ylabel(r"$\Delta$ Rating Contribution")
    axes[1, 1].axhline(0, color="gray", linestyle="--", alpha=0.7)

    plt.tight_layout()
    p1 = out_dir / "extrinsic_impact_curves.png"
    plt.savefig(p1, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 2: Venue Regularized Shrinkage Effects
    # -------------------------------------------------------------
    venue_df = pd.DataFrame(
        {
            "venue": res["venue_levels"],
            "beta": res["venue_betas"],
            "count": [(df["venue"] == v).sum() for v in res["venue_levels"]],
        }
    )
    # Filter to top 15 positive and bottom 15 negative venues
    venue_top_bot = (
        pd.concat(
            [
                venue_df.sort_values("beta", ascending=False).head(12),
                venue_df.sort_values("beta", ascending=True).head(12),
            ]
        )
        .drop_duplicates()
        .sort_values("beta", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["#d62728" if b < 0 else "#2ca02c" for b in venue_top_bot["beta"]]
    bars = ax.barh(
        venue_top_bot["venue"], venue_top_bot["beta"], color=colors, alpha=0.85
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Regularized Venue Atmosphere Effects (Shrinkage Controlled)")
    ax.set_xlabel(r"Estimated Venue Bonus / Drag ($\Delta$ Rating)")

    # Label sample sizes on bars
    for bar, (_, row) in zip(bars, venue_top_bot.iterrows()):
        x_pos = bar.get_width()
        ha = "left" if x_pos >= 0 else "right"
        offset = 0.005 if x_pos >= 0 else -0.005
        ax.text(
            x_pos + offset,
            bar.get_y() + bar.get_height() / 2,
            f" n={row['count']}",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )

    plt.tight_layout()
    p2 = out_dir / "venue_shrinkage_effects.png"
    plt.savefig(p2, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 3: Intrinsic PCA Feature Space Landscape
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(
        df["int_PC1"],
        df["int_PC2"],
        c=df["rating_extrinsic_neutral"],
        cmap="viridis",
        alpha=0.6,
        s=25,
        edgecolor="none",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Extrinsic-Neutralized Rating")

    # Annotate high density or top rated beers in the landscape
    top_gems = (
        df.sort_values("rating_extrinsic_neutral", ascending=False)
        .drop_duplicates("beer_name")
        .head(8)
    )
    for _, row in top_gems.iterrows():
        ax.annotate(
            row["beer_name"][:22],
            (row["int_PC1"], row["int_PC2"]),
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8),
        )

    ax.set_title(
        f"Intrinsic PCA Landscape (PC1: {res['int_pca']['var_explained'][0]*100:.1f}% var, "
        f"PC2: {res['int_pca']['var_explained'][1]*100:.1f}% var)"
    )
    ax.set_xlabel("Intrinsic PC1 (Roasty / Gravity / Malt / Body)")
    ax.set_ylabel("Intrinsic PC2 (Hops / Tartness / Freshness)")
    plt.tight_layout()
    p3 = out_dir / "intrinsic_pca_landscape.png"
    plt.savefig(p3, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 4: Rating Disambiguation (Raw Discrete vs Neutral Continuous)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9))
    # Filter to raw 4.25, 4.5, 4.75, 5.0 to see the disambiguation explicitly
    high_df = df[df["rating_raw"] >= 4.0].copy()
    sns.stripplot(
        data=high_df,
        x="rating_raw",
        y="rating_extrinsic_neutral",
        jitter=0.25,
        alpha=0.4,
        palette="magma",
        size=5,
        ax=ax,
    )
    ax.set_title(
        "Disambiguating Ratings: Raw Discrete Untappd vs Continuous Extrinsic-Neutral"
    )
    ax.set_xlabel("Original Raw Untappd Rating Score")
    ax.set_ylabel("Extrinsic-Neutral Continuous Rating Score")

    # Add reference diagonal
    unique_raws = sorted(high_df["rating_raw"].unique())
    ax.plot(
        range(len(unique_raws)),
        unique_raws,
        color="red",
        linestyle="--",
        label="Raw = Neutral Identity",
    )
    ax.legend(loc="upper left")

    plt.tight_layout()
    p4 = out_dir / "rating_disambiguation_scatter.png"
    plt.savefig(p4, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 5: Intrinsic Value Above Replacement (Style-Normalized Distribution)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))
    top_styles = df["superstyle"].value_counts().head(6).index
    sns.boxplot(
        data=df[df["superstyle"].isin(top_styles)],
        x="superstyle",
        y="rating_style_normalized",
        palette="Set2",
        ax=ax,
    )
    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_title("Intrinsic Value-Above-Replacement by Style (Best-In-Class Outliers)")
    ax.set_xlabel("Style Category")
    ax.set_ylabel("Rating Above Style Expectation")
    plt.tight_layout()
    p5 = out_dir / "style_normalized_boxplots.png"
    plt.savefig(p5, dpi=300)
    plt.close(fig)

    return [p1, p2, p3, p4, p5]


if __name__ == "__main__":
    print("Loading checkins and fitting decomposition...")
    df, res = get_analysis()
    print(
        f"Fit complete. Analyzed {len(df)} checkins across {df['beer_name'].nunique()} beers."
    )
    print(f"Model R^2: {res['r_squared']:.4f}")
    print(f"Global rating mean mu: {res['mu']:.3f}")
    print("\nTop 10 Beers (Extrinsic-Neutralized, Absolute Enjoyment):")
    best_neutral = (
        df.groupby(["beer_name", "brewery", "superstyle", "abv"])[
            "rating_extrinsic_neutral"
        ]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(10)
    )
    print(best_neutral.to_string(index=False))

    print("\nTop 10 Beers (Style-Normalized, Value-Above-Replacement Outliers):")
    best_style = (
        df.groupby(["beer_name", "brewery", "superstyle", "abv"])[
            "rating_style_normalized"
        ]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(10)
    )
    print(best_style.to_string(index=False))

    print("\nGenerating diagnostic plots in scripts/slop/out/...")
    plots = generate_plots(df, res)
    for p in plots:
        print(f"Saved: {p}")
