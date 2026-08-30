from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn

import untappd
import untappd_utils

MAX_SESSION_GAP_HOURS = 6

ABV_BINS = [
    ("x < 4%", 4.0),
    ("4% <= x < 6%", 6.0),
    ("6% <= x < 8%", 8.0),
    ("8% <= x < 10%", 10.0),
    ("10% <= x < 12%", 12.0),
    ("x >= 12%", float("inf")),
]


def categorize_abv(abv: float) -> str:
    for label, upper in ABV_BINS:
        if abv < upper:
            return label
    return ABV_BINS[-1][0]


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


def compute_markov(
    sessions: list[list[untappd.Checkin]],
    bin_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(bin_labels)
    bin_indices = {label: i for i, label in enumerate(bin_labels)}
    counts = np.zeros((n, n + 1))
    marginal = np.zeros(n)

    for session in sessions:
        indices = [bin_indices[categorize_abv(c.beer.abv or 0.0)] for c in session]
        for idx in indices:
            marginal[idx] += 1
        for i in range(len(indices) - 1):
            counts[indices[i], indices[i + 1]] += 1
        counts[indices[-1], -1] += 1

    p_full = counts / counts.sum(axis=1, keepdims=True)

    cont_counts = counts[:, :-1]
    p_cont = cont_counts / cont_counts.sum(axis=1, keepdims=True)

    evals, evecs = np.linalg.eig(p_cont.T)
    pi_stat = np.real(evecs[:, int(np.argmin(np.abs(evals - 1.0)))])
    pi_stat = pi_stat / pi_stat.sum()
    pi_marg = marginal / marginal.sum()

    return p_full, pi_stat, pi_marg


@untappd_utils.show_or_save_to_out_file
def plot_markov_abv(
    bin_labels: list[str],
    p_full: np.ndarray,
    pi_stat: np.ndarray,
    pi_marg: np.ndarray,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    col_labels = bin_labels + ["Session End"]

    seaborn.heatmap(
        p_full,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=col_labels,
        yticklabels=bin_labels,
        cbar_kws={"label": "P(Next | Current)"},
        ax=ax1,
        linewidths=0.5,
        linecolor="#e0e0e0",
    )
    ax1.axvline(
        len(bin_labels),
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
    )
    ax1.set_title(
        "ABV Markov Transition Matrix P(Next | Current)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("Next State", fontsize=10)
    ax1.set_ylabel("Current ABV Bracket", fontsize=10)
    ax1.set_xticklabels(col_labels, rotation=30, ha="right")
    ax1.set_yticklabels(bin_labels, rotation=0)

    y_pos = np.arange(len(bin_labels))
    bar_width = 0.35

    bars1 = ax2.barh(
        y_pos - bar_width / 2,
        pi_marg * 100,
        bar_width,
        label="Marginal (Overall %)",
        color="#4C72B0",
    )
    bars2 = ax2.barh(
        y_pos + bar_width / 2,
        pi_stat * 100,
        bar_width,
        label="Stationary (Steady-State %)",
        color="#DD8452",
    )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(bin_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Probability (%)", fontsize=10)
    ax2.set_title(
        "Stationary vs Marginal ABV Distribution",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax2.legend(loc="lower right")

    for bar in bars1:
        w = bar.get_width()
        ax2.text(
            w + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="#2b4570",
        )

    for bar in bars2:
        w = bar.get_width()
        ax2.text(
            w + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="#a04818",
        )

    ax2.set_xlim(0, max(np.max(pi_marg), np.max(pi_stat)) * 100 + 5)
    plt.tight_layout()


def main(
    out_file: Optional[Union[Path, str]] = (
        Path(__file__).resolve().parent / "out" / "markov_abv.png"
    ),
) -> None:
    checkins = untappd.load_latest_checkins()
    sessions = segment_sessions(checkins)
    bin_labels = [label for label, _ in ABV_BINS]
    p_full, pi_stat, pi_marg = compute_markov(sessions, bin_labels)

    plot_markov_abv(
        bin_labels,
        p_full,
        pi_stat,
        pi_marg,
        out_file=out_file,
    )


if __name__ == "__main__":
    main()
