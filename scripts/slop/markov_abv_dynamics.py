from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

import untappd
import untappd_utils

MAX_SESSION_GAP_HOURS = 6


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


def compute_drift_curve(
    transitions: list[tuple[float, float]],
    bin_width: float = 1.0,
    min_abv: float = 2.0,
    max_abv: float = 14.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.arange(min_abv + bin_width / 2, max_abv, bin_width)
    means = []
    p25 = []
    p75 = []
    counts = []

    for c in centers:
        low, high = c - bin_width / 2, c + bin_width / 2
        deltas = [d for a, d in transitions if low <= a < high]
        if deltas:
            means.append(float(np.mean(deltas)))
            p25.append(float(np.percentile(deltas, 25)))
            p75.append(float(np.percentile(deltas, 75)))
            counts.append(len(deltas))
        else:
            means.append(np.nan)
            p25.append(np.nan)
            p75.append(np.nan)
            counts.append(0)

    return centers, np.array(means), np.array(p25), np.array(p75), np.array(counts)


def compute_directional_probabilities(
    sessions: list[list[untappd.Checkin]],
    bins: list[tuple[float, float, str]],
    tolerance: float = 0.25,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = [b[2] for b in bins]
    n = len(bins)
    p_up = np.zeros(n)
    p_same = np.zeros(n)
    p_down = np.zeros(n)
    p_end = np.zeros(n)
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
                p_end[bin_idx] += 1
            else:
                next_abv = session[i + 1].beer.abv or 0.0
                delta = next_abv - abv
                if delta > tolerance:
                    p_up[bin_idx] += 1
                elif delta < -tolerance:
                    p_down[bin_idx] += 1
                else:
                    p_same[bin_idx] += 1

    valid = totals > 0
    p_up = np.divide(p_up, totals, out=np.zeros_like(p_up), where=valid)
    p_same = np.divide(p_same, totals, out=np.zeros_like(p_same), where=valid)
    p_down = np.divide(p_down, totals, out=np.zeros_like(p_down), where=valid)
    p_end = np.divide(p_end, totals, out=np.zeros_like(p_end), where=valid)

    return labels, p_up, p_same, p_down, p_end, totals


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
    drift_means: np.ndarray,
    drift_p25: np.ndarray,
    drift_p75: np.ndarray,
    linear_fit: tuple[float, float, float],
    cat_labels: list[str],
    p_up: np.ndarray,
    p_same: np.ndarray,
    p_down: np.ndarray,
    p_end: np.ndarray,
    drinks: np.ndarray,
    drink_means: np.ndarray,
    drink_medians: np.ndarray,
    drink_counts: np.ndarray,
) -> None:
    slope, intercept, attractor = linear_fit

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    # --- Panel 1: Local Dynamical Drift Function mu(x) ---
    ax1 = axes[0]
    ax1.axhline(0, color="#333333", linestyle="--", linewidth=1.2, alpha=0.8)

    # Linear drift regression fit
    x_line = np.linspace(2.5, 13.5, 100)
    y_line = slope * x_line + intercept
    ax1.plot(
        x_line,
        y_line,
        color="#888888",
        linestyle=":",
        linewidth=1.8,
        label=f"Linear Fit (slope={slope:.2f}/%)",
    )

    # Shaded IQR band and mean curve
    valid_mask = ~np.isnan(drift_means)
    ax1.fill_between(
        centers[valid_mask],
        drift_p25[valid_mask],
        drift_p75[valid_mask],
        color="#4C72B0",
        alpha=0.18,
        label="Interquartile Range (25th–75th %ile)",
    )
    ax1.plot(
        centers[valid_mask],
        drift_means[valid_mask],
        marker="o",
        color="#4C72B0",
        linewidth=2.5,
        markersize=6,
        label=r"Empirical Drift $\mu(\mathrm{ABV}) = \mathbb{E}[\Delta \mathrm{ABV}]$",
    )

    # Mark Attractor / Equilibrium
    ax1.axvline(attractor, color="#d62728", linestyle="--", linewidth=1.4, alpha=0.9)
    ax1.scatter(
        [attractor],
        [0],
        color="#d62728",
        s=80,
        zorder=5,
        label=f"Attractor $x^* = {attractor:.2f}\\%$",
    )
    ax1.annotate(
        f"Equilibrium Point: {attractor:.2f}%\n"
        r"($\Delta \mathrm{ABV} > 0$ below, $< 0$ above)",
        xy=(attractor, 0),
        xytext=(attractor + 0.6, 1.2),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        color="#901a1e",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffebeb", ec="#d62728", alpha=0.8),
    )

    ax1.set_title(
        r"1. Local Dynamical Drift $\mu(\mathrm{ABV}) = \mathbb{E}[\Delta \mathrm{ABV} \mid \mathrm{ABV}_t]$",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel(r"Current ABV $x_t$ (%)", fontsize=10)
    ax1.set_ylabel(r"Expected Change $\Delta \mathrm{ABV}$ (% pts)", fontsize=10)
    ax1.set_xlim(2.0, 14.0)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=8.5)

    # --- Panel 2: Directional Probabilities + Termination Hazard ---
    ax2 = axes[1]
    x_cat = np.arange(len(cat_labels))
    bar_width = 0.65

    # Stacked bars: Step Up, Step Same, Step Down, Session End
    bars_up = ax2.bar(
        x_cat,
        p_up * 100,
        bar_width,
        label=r"Step Up ($\Delta > +0.25\%$)",
        color="#2ca02c",
        alpha=0.85,
    )
    bars_same = ax2.bar(
        x_cat,
        p_same * 100,
        bar_width,
        bottom=p_up * 100,
        label=r"Same ($\pm 0.25\%$)",
        color="#1f77b4",
        alpha=0.85,
    )
    bars_down = ax2.bar(
        x_cat,
        p_down * 100,
        bar_width,
        bottom=(p_up + p_same) * 100,
        label=r"Step Down ($\Delta < -0.25\%$)",
        color="#ff7f0e",
        alpha=0.85,
    )
    bars_end = ax2.bar(
        x_cat,
        p_end * 100,
        bar_width,
        bottom=(p_up + p_same + p_down) * 100,
        label="Session End (Absorbing)",
        color="#d62728",
        alpha=0.85,
    )

    ax2.set_xticks(x_cat)
    ax2.set_xticklabels(cat_labels, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Outcome Probability (%)", fontsize=10)
    ax2.set_xlabel("Current ABV Bracket", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title(
        "2. Directional Transition & Termination Hazard",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    # Annotate step up vs step down dominance
    for idx, (u, d) in enumerate(zip(p_up, p_down)):
        if u > 0.45:
            ax2.text(
                idx,
                u * 50,
                f"{u:.0%}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=8,
            )
        if d > 0.45:
            ax2.text(
                idx,
                (p_up[idx] + p_same[idx] + d * 0.5) * 100,
                f"{d:.0%}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=8,
            )

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

    # Add count badges above markers
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

    # Overall escalation rate per drink
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
        "Untappd ABV Dynamical Transitions & Directionality Analysis",
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

    # Extract transitions
    valid_transitions: list[tuple[float, float]] = []
    for s in sessions:
        for i in range(len(s) - 1):
            a1 = s[i].beer.abv
            a2 = s[i + 1].beer.abv
            if a1 is not None and a2 is not None and a1 > 0 and a2 > 0:
                valid_transitions.append((a1, a2 - a1))

    # 1. Continuous drift curve
    centers, means, p25, p75, counts = compute_drift_curve(valid_transitions)

    # Linear fit between 3% and 12% to find the equilibrium attractor
    fit_pairs = [(a, d) for a, d in valid_transitions if 3.0 <= a <= 12.0]
    fit_x = np.array([p[0] for p in fit_pairs])
    fit_y = np.array([p[1] for p in fit_pairs])
    slope, intercept = np.polyfit(fit_x, fit_y, 1)
    attractor = float(-intercept / slope)

    # 2. Directional probabilities & termination hazard
    bins = [
        (0.0, 4.0, "<4%"),
        (4.0, 5.0, "4-5%"),
        (5.0, 6.0, "5-6%"),
        (6.0, 7.0, "6-7%"),
        (7.0, 8.0, "7-8%"),
        (8.0, 9.0, "8-9%"),
        (9.0, 10.0, "9-10%"),
        (10.0, 12.0, "10-12%"),
        (12.0, 30.0, ">=12%"),
    ]
    cat_labels, p_up, p_same, p_down, p_end, totals = compute_directional_probabilities(
        sessions, bins
    )

    # 3. Progression by drink number
    drinks, drink_means, drink_medians, drink_counts = compute_session_escalation(
        sessions, max_drinks=7
    )

    print("=" * 65)
    print("ABV DYNAMICAL DIRECTIONALITY ANALYSIS")
    print("=" * 65)
    print(
        f"Analyzed {len(valid_transitions)} transitions across {len(sessions)} sessions."
    )
    print(f"Dynamical Attractor (Equilibrium ABV): {attractor:.2f}%")
    print(f"Mean Escalation Rate: +{slope * -1:.2f}% reversion per % ABV above base")
    print("-" * 65)
    print("Session Progression by Drink Index:")
    for d, m, med, cnt in zip(drinks, drink_means, drink_medians, drink_counts):
        print(f"  Beer #{d}: Mean = {m:5.2f}%, Median = {med:5.2f}% (n={cnt:4d})")

    plot_abv_dynamics(
        centers,
        means,
        p25,
        p75,
        (float(slope), float(intercept), attractor),
        cat_labels,
        p_up,
        p_same,
        p_down,
        p_end,
        drinks,
        drink_means,
        drink_medians,
        drink_counts,
        out_file=out_file,
    )


if __name__ == "__main__":
    main()
