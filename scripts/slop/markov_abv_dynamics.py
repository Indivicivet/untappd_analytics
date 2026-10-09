from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

import untappd
import untappd_utils

MAX_SESSION_GAP_HOURS = 6

# Brackets for relative transitions (r = next_abv / current_abv)
# 1. Big Step Down: r < 0.65 (e.g. 11% -> <7.1%, 5% -> <3.2%)
# 2. Modest Step Down: 0.65 <= r < 0.88 (e.g. 11% -> 7.2-9.6%, 5% -> 3.3-4.4%)
# 3. Comparable / Similar: 0.88 <= r <= 1.14 (+- ~12-14% relative)
# 4. Modest Step Up: 1.14 < r <= 1.50 (e.g. 5% -> 5.7-7.5%, 8% -> 9.1-12%)
# 5. Big Step Up: r > 1.50 (e.g. 4% -> >6%, 5% -> >7.5%)
# 6. Session End: absorbing state
CATEGORICAL_BINS = [
    (0.0, 3.5, "<3.5%"),
    (3.5, 4.5, "3.5-4.5%"),
    (4.5, 5.5, "4.5-5.5%"),
    (5.5, 6.5, "5.5-6.5%"),
    (6.5, 7.5, "6.5-7.5%"),
    (7.5, 8.5, "7.5-8.5%"),
    (8.5, 10.0, "8.5-10%"),
    (10.0, 12.0, "10-12%"),
    (12.0, 30.0, ">=12%"),
]


def segment_sessions(
    checkins: list[untappd.Checkin],
    max_gap: timedelta = timedelta(hours=MAX_SESSION_GAP_HOURS),
) -> list[list[untappd.Checkin]]:
    sessions: list[list[untappd.Checkin]] = []
    current_session: list[untappd.Checkin] = []

    for c in checkins:
        if (
            current_session
            and c.datetime
            and current_session[-1].datetime
            and (c.datetime - current_session[-1].datetime) > max_gap
        ):
            sessions.append(current_session)
            current_session = []
        current_session.append(c)

    if current_session:
        sessions.append(current_session)
    return sessions


def compute_relative_drift_curve(
    transitions: list[tuple[float, float, float]],
    bin_width: float = 1.0,
    min_abv: float = 2.0,
    max_abv: float = 14.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.arange(min_abv + bin_width / 2, max_abv, bin_width)
    means_pct = []
    medians_pct = []
    p25_pct = []
    p75_pct = []
    counts = []

    for c in centers:
        low, high = c - bin_width / 2, c + bin_width / 2
        rel_deltas = [rel * 100 for a1, _, rel in transitions if low <= a1 < high]
        if rel_deltas:
            means_pct.append(float(np.mean(rel_deltas)))
            medians_pct.append(float(np.median(rel_deltas)))
            p25_pct.append(float(np.percentile(rel_deltas, 25)))
            p75_pct.append(float(np.percentile(rel_deltas, 75)))
            counts.append(len(rel_deltas))
        else:
            means_pct.append(np.nan)
            medians_pct.append(np.nan)
            p25_pct.append(np.nan)
            p75_pct.append(np.nan)
            counts.append(0)

    return (
        centers,
        np.array(means_pct),
        np.array(medians_pct),
        np.array(p25_pct),
        np.array(p75_pct),
        np.array(counts),
    )


def compute_fine_relative_outcomes(
    sessions: list[list[untappd.Checkin]],
    bins: list[tuple[float, float, str]],
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    labels = [b[2] for b in bins]
    n = len(bins)

    outcomes = {
        "big_down": np.zeros(n),
        "mod_down": np.zeros(n),
        "similar": np.zeros(n),
        "mod_up": np.zeros(n),
        "big_up": np.zeros(n),
        "end": np.zeros(n),
    }
    totals = np.zeros(n)

    for session in sessions:
        for i, c in enumerate(session):
            abv = c.beer.abv or 0.0
            bin_idx = None
            for b_i, (b_low, b_high, _) in enumerate(bins):
                if b_low <= abv < b_high:
                    bin_idx = b_i
                    break
            if bin_idx is None:
                continue

            totals[bin_idx] += 1
            if i == len(session) - 1:
                outcomes["end"][bin_idx] += 1
            else:
                next_abv = session[i + 1].beer.abv or 0.0
                base_abv = max(abv, 0.5)
                ratio = next_abv / base_abv
                if ratio < 0.65:
                    outcomes["big_down"][bin_idx] += 1
                elif ratio < 0.88:
                    outcomes["mod_down"][bin_idx] += 1
                elif ratio <= 1.14:
                    outcomes["similar"][bin_idx] += 1
                elif ratio <= 1.50:
                    outcomes["mod_up"][bin_idx] += 1
                else:
                    outcomes["big_up"][bin_idx] += 1

    valid = totals > 0
    probs = {
        k: np.divide(v, totals, out=np.zeros_like(v), where=valid)
        for k, v in outcomes.items()
    }
    return labels, probs, totals


def compute_session_escalation(
    sessions: list[list[untappd.Checkin]],
    max_drinks: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drinks = np.arange(1, max_drinks + 1)
    means = []
    medians = []
    counts = []

    for d in drinks:
        abvs = [
            s[d - 1].beer.abv
            for s in sessions
            if len(s) >= d and s[d - 1].beer.abv is not None and s[d - 1].beer.abv > 0
        ]
        if abvs:
            means.append(float(np.mean(abvs)))
            medians.append(float(np.median(abvs)))
            counts.append(len(abvs))
        else:
            means.append(np.nan)
            medians.append(np.nan)
            counts.append(0)

    return drinks, np.array(means), np.array(medians), np.array(counts)


@untappd_utils.show_or_save_to_out_file
def plot_abv_dynamics(
    centers: np.ndarray,
    drift_means_pct: np.ndarray,
    drift_medians_pct: np.ndarray,
    drift_p25_pct: np.ndarray,
    drift_p75_pct: np.ndarray,
    attractor: float,
    cat_labels: list[str],
    probs: dict[str, np.ndarray],
    drinks: np.ndarray,
    drink_means: np.ndarray,
    drink_medians: np.ndarray,
    drink_counts: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.8))

    # --- Panel 1: Relative Local Dynamical Drift mu_rel(x) ---
    ax1 = axes[0]
    ax1.axhline(0, color="#333333", linestyle="--", linewidth=1.2, alpha=0.8)

    valid_mask = ~np.isnan(drift_means_pct)

    # Shaded IQR band
    ax1.fill_between(
        centers[valid_mask],
        drift_p25_pct[valid_mask],
        drift_p75_pct[valid_mask],
        color="#4C72B0",
        alpha=0.18,
        label="Interquartile Range (25th–75th %ile)",
    )

    # Mean relative change
    ax1.plot(
        centers[valid_mask],
        drift_means_pct[valid_mask],
        marker="o",
        color="#2b5c8f",
        linewidth=2.2,
        markersize=5.5,
        label=r"Mean Rel. Change $\mathbb{E}[(\mathrm{ABV}_{t+1}-\mathrm{ABV}_t)/\mathrm{ABV}_t]$",
    )

    # Median relative change
    ax1.plot(
        centers[valid_mask],
        drift_medians_pct[valid_mask],
        marker="^",
        color="#e66101",
        linewidth=2.0,
        linestyle="--",
        markersize=5.5,
        label="Median Rel. Change",
    )

    # Attractor vertical line
    ax1.axvline(attractor, color="#d62728", linestyle="--", linewidth=1.4, alpha=0.9)
    ax1.scatter([attractor], [0], color="#d62728", s=80, zorder=5)
    ax1.annotate(
        f"Equilibrium Attractor: {attractor:.2f}%\n" r"(Net relative drift = 0%)",
        xy=(attractor, 0),
        xytext=(attractor + 0.5, 25),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        color="#901a1e",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffebeb", ec="#d62728", alpha=0.85),
    )

    ax1.set_title(
        "1. Relative ABV Dynamical Drift (Scale-Invariant)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("Current ABV Bracket (%)", fontsize=10)
    ax1.set_ylabel("Expected Relative Change (%)", fontsize=10)
    ax1.set_xlim(2.0, 14.0)
    ax1.set_ylim(-65, 85)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5)

    # --- Panel 2: Granular Relative Transition Outcomes ---
    ax2 = axes[1]
    x_cat = np.arange(len(cat_labels))
    bar_width = 0.68

    # Stack segments with intuitive progression colors
    # Big Step Up -> Modest Step Up -> Similar -> Modest Step Down -> Big Step Down -> Session End
    palette = [
        ("big_up", "Big Step Up (> +50%)", "#1a9641"),
        ("mod_up", "Modest Step Up (+14% to +50%)", "#a6d96a"),
        ("similar", "Similar (±13% relative)", "#4393c3"),
        ("mod_down", "Modest Step Down (-12% to -35%)", "#fdae61"),
        ("big_down", "Big Step Down (< -35%)", "#d7191c"),
        ("end", "Session End (Absorbing)", "#636363"),
    ]

    bottom = np.zeros(len(cat_labels))
    for key, label_text, color in palette:
        vals = probs[key] * 100
        bars = ax2.bar(
            x_cat,
            vals,
            bar_width,
            bottom=bottom,
            label=label_text,
            color=color,
            alpha=0.92,
            edgecolor="#ffffff",
            linewidth=0.5,
        )
        # Add labels to prominent segments
        for idx, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 16.0:
                ax2.text(
                    idx,
                    b + v / 2,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if key in ("big_up", "big_down", "end", "similar")
                        else "#222222"
                    ),
                    fontweight="bold",
                    fontsize=8.0,
                )
        bottom += vals

    ax2.set_xticks(x_cat)
    ax2.set_xticklabels(cat_labels, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Outcome Distribution (%)", fontsize=10)
    ax2.set_xlabel("Current ABV Bracket", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title(
        "2. Relative Transition Spectra & Termination Hazard",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8.0)

    # --- Panel 3: Session Trajectory ABV(k) by Drink Number ---
    ax3 = axes[2]
    ax3.plot(
        drinks,
        drink_means,
        marker="s",
        color="#1f77b4",
        linewidth=2.2,
        label="Mean ABV",
    )
    ax3.plot(
        drinks,
        drink_medians,
        marker="^",
        color="#ff7f0e",
        linewidth=2.0,
        linestyle="--",
        label="Median ABV",
    )

    for d, m, c in zip(drinks, drink_means, drink_counts):
        ax3.text(
            d,
            m + 0.12,
            f"{m:.2f}%\n(n={c})",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f77b4",
        )

    d_slope, d_intercept = np.polyfit(drinks, drink_means, 1)
    x_seq = np.linspace(1, max(drinks), 50)
    ax3.plot(
        x_seq,
        d_slope * x_seq + d_intercept,
        color="#555555",
        linestyle=":",
        linewidth=1.5,
        label=f"Escalation Trend (+{d_slope:.2f}%/drink)",
    )

    ax3.set_title(
        r"3. Session Progression $\mathbb{E}[\mathrm{ABV}_k \mid \mathrm{Length} \geq k]$",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax3.set_xlabel("Drink Index in Session (k)", fontsize=10)
    ax3.set_ylabel("ABV (%)", fontsize=10)
    ax3.set_xticks(drinks)
    ax3.set_ylim(4.5, 7.8)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", fontsize=8.5)

    plt.suptitle(
        "Untappd ABV Dynamical Transitions & Relative Directionality",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()


def main(
    out_file: Optional[Union[Path, str]] = (
        Path(__file__).resolve().parent / "out" / "markov_abv_dynamics.png"
    ),
) -> None:
    checkins = untappd.load_latest_checkins()
    sessions = segment_sessions(checkins)

    # Extract transitions with relative metrics (avoiding div by zero by clamping denominator >= 0.5%)
    valid_transitions: list[tuple[float, float, float]] = []
    for s in sessions:
        for i in range(len(s) - 1):
            a1 = s[i].beer.abv
            a2 = s[i + 1].beer.abv
            if a1 is not None and a2 is not None and a1 > 0 and a2 > 0:
                base = max(a1, 0.5)
                rel_delta = (a2 - a1) / base
                valid_transitions.append((a1, a2, rel_delta))

    # 1. Continuous relative drift curve
    centers, means_pct, medians_pct, p25_pct, p75_pct, counts = (
        compute_relative_drift_curve(valid_transitions)
    )

    # Equilibrium attractor: point where median/mean relative drift crosses 0
    # Between 5% and 8% ABV:
    fit_pairs = [(a1, rel) for a1, _, rel in valid_transitions if 4.0 <= a1 <= 9.0]
    fit_x = np.array([p[0] for p in fit_pairs])
    fit_y = np.array([p[1] for p in fit_pairs])
    slope, intercept = np.polyfit(fit_x, fit_y, 1)
    attractor = float(-intercept / slope)

    # 2. Granular relative transition outcomes + termination hazard
    cat_labels, probs, totals = compute_fine_relative_outcomes(
        sessions, CATEGORICAL_BINS
    )

    # 3. Session progression
    drinks, drink_means, drink_medians, drink_counts = compute_session_escalation(
        sessions, max_drinks=7
    )

    print("=" * 70)
    print("RELATIVE ABV DYNAMICAL DIRECTIONALITY ANALYSIS")
    print("=" * 70)
    print(
        f"Analyzed {len(valid_transitions)} transitions across {len(sessions)} sessions."
    )
    print(f"Dynamical Attractor (Zero Relative Drift): {attractor:.2f}% ABV")
    print("-" * 70)
    print("Relative Outcome Distribution by Current ABV:")
    print(
        f"{'Bracket':10s} | {'BigUp':>6s} | {'ModUp':>6s} | {'Similar':>7s} | {'ModDown':>7s} | {'BigDown':>7s} | {'End':>6s}"
    )
    for i, lbl in enumerate(cat_labels):
        print(
            f"{lbl:10s} | "
            f"{probs['big_up'][i]:5.1%} | "
            f"{probs['mod_up'][i]:5.1%} | "
            f"{probs['similar'][i]:6.1%} | "
            f"{probs['mod_down'][i]:6.1%} | "
            f"{probs['big_down'][i]:6.1%} | "
            f"{probs['end'][i]:5.1%}"
        )

    plot_abv_dynamics(
        centers,
        means_pct,
        medians_pct,
        p25_pct,
        p75_pct,
        attractor,
        cat_labels,
        probs,
        drinks,
        drink_means,
        drink_medians,
        drink_counts,
        out_file=out_file,
    )


if __name__ == "__main__":
    main()
