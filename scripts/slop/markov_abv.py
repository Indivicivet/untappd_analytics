from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn

import untappd
import untappd_utils

MAX_SESSION_GAP_HOURS = 6

COARSE_BINS = [
    ("x < 4%", 4.0),
    ("4% <= x < 6%", 6.0),
    ("6% <= x < 8%", 8.0),
    ("8% <= x < 10%", 10.0),
    ("10% <= x < 12%", 12.0),
    ("x >= 12%", float("inf")),
]

FINE_BINS = [
    ("x < 0.51%", 0.51),
    ("0.51% <= x < 3%", 3.0),
    ("3% <= x < 4%", 4.0),
    ("4% <= x < 5%", 5.0),
    ("5% <= x < 6%", 6.0),
    ("6% <= x < 7%", 7.0),
    ("7% <= x < 8%", 8.0),
    ("8% <= x < 9%", 9.0),
    ("9% <= x < 10%", 10.0),
    ("10% <= x < 11%", 11.0),
    ("11% <= x < 12%", 12.0),
    ("12% <= x < 14%", 14.0),
    ("14% <= x < 16%", 16.0),
    ("x >= 16%", float("inf")),
]


def categorize_abv(abv: float, bins: list[tuple[str, float]]) -> str:
    for label, upper in bins:
        if abv < upper:
            return label
    return bins[-1][0]


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
    bins: list[tuple[str, float]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bin_labels = [label for label, _ in bins]
    n = len(bin_labels)
    bin_indices = {label: i for i, label in enumerate(bin_labels)}
    counts = np.zeros((n, n + 1))
    marginal = np.zeros(n)

    for session in sessions:
        indices = [
            bin_indices[categorize_abv(c.beer.abv or 0.0, bins)] for c in session
        ]
        for idx in indices:
            marginal[idx] += 1
        for i in range(len(indices) - 1):
            counts[indices[i], indices[i + 1]] += 1
        counts[indices[-1], -1] += 1

    p_full = np.divide(
        counts,
        counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(counts),
        where=counts.sum(axis=1, keepdims=True) > 0,
    )

    q_base = counts.sum(axis=0, keepdims=True) / counts.sum()
    p_rel = np.divide(
        p_full,
        q_base,
        out=np.zeros_like(p_full),
        where=q_base > 0,
    )

    cont_counts = counts[:, :-1]
    p_cont = np.divide(
        cont_counts,
        cont_counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(cont_counts),
        where=cont_counts.sum(axis=1, keepdims=True) > 0,
    )

    evals, evecs = np.linalg.eig(p_cont.T)
    pi_stat = np.real(evecs[:, int(np.argmin(np.abs(evals - 1.0)))])
    if pi_stat.sum() > 0:
        pi_stat = pi_stat / pi_stat.sum()
    else:
        pi_stat = np.zeros(n)

    pi_marg = marginal / marginal.sum()
    pi_ratio = np.divide(
        pi_stat,
        pi_marg,
        out=np.zeros_like(pi_stat),
        where=pi_marg > 0,
    )

    return bin_labels, p_full, p_rel, pi_stat, pi_marg, pi_ratio


@untappd_utils.show_or_save_to_out_file
def plot_markov_abv(
    bin_labels: list[str],
    p_matrix: np.ndarray,
    pi_stat: np.ndarray,
    pi_marg: np.ndarray,
    pi_ratio: Optional[np.ndarray] = None,
    relative: bool = False,
    title_suffix: str = "",
) -> None:
    is_fine = len(bin_labels) > 10
    figsize = (20, 10) if is_fine else (16, 6.5)
    font_size = 8.5 if is_fine else 9.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    col_labels = bin_labels + ["Session End"]

    if relative:
        vmax = max(2.0, float(np.percentile(p_matrix, 98)))
        seaborn.heatmap(
            p_matrix,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=1.0,
            vmin=0.0,
            vmax=vmax,
            xticklabels=col_labels,
            yticklabels=bin_labels,
            cbar_kws={"label": "Relative Probability (Lift vs Marginal Base Rate)"},
            ax=ax1,
            linewidths=0.5,
            linecolor="#e0e0e0",
            annot_kws={"size": font_size - 0.5},
        )
        ax1.set_title(
            f"Relative ABV Transition Probability (Lift = P / Base Rate){title_suffix}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
    else:
        seaborn.heatmap(
            p_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=col_labels,
            yticklabels=bin_labels,
            cbar_kws={"label": "P(Next | Current)"},
            ax=ax1,
            linewidths=0.5,
            linecolor="#e0e0e0",
            annot_kws={"size": font_size - 0.5},
        )
        ax1.set_title(
            f"ABV Markov Transition Matrix P(Next | Current){title_suffix}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

    ax1.axvline(
        len(bin_labels),
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
    )
    ax1.set_xlabel("Next State", fontsize=10)
    ax1.set_ylabel("Current ABV Bracket", fontsize=10)
    ax1.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=font_size)
    ax1.set_yticklabels(bin_labels, rotation=0, fontsize=font_size)

    y_pos = np.arange(len(bin_labels))

    if relative and pi_ratio is not None:
        colors = ["#DD8452" if r >= 1.0 else "#4C72B0" for r in pi_ratio]
        bars = ax2.barh(
            y_pos,
            pi_ratio,
            height=0.5,
            color=colors,
            edgecolor="#555555",
            linewidth=0.5,
        )
        ax2.axvline(
            1.0,
            color="#333333",
            linewidth=1.2,
            linestyle="--",
            label="Neutral (1.0x Baseline)",
        )
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(bin_labels, fontsize=font_size)
        ax2.invert_yaxis()
        ax2.set_xlabel("Ratio (Stationary % / Marginal %)", fontsize=10)
        ax2.set_title(
            f"Within-Session ABV Enrichment Ratio (Steady-State / Overall){title_suffix}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax2.legend(loc="lower right")

        for bar in bars:
            w = bar.get_width()
            ax2.text(
                w + 0.02 if w >= 0.5 else 0.52,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}x",
                va="center",
                ha="left",
                fontsize=font_size,
                fontweight="bold" if abs(w - 1.0) > 0.15 else "normal",
            )
        ax2.set_xlim(0, max(np.max(pi_ratio) + 0.25, 1.4))
    else:
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
        ax2.set_yticklabels(bin_labels, fontsize=font_size)
        ax2.invert_yaxis()
        ax2.set_xlabel("Probability (%)", fontsize=10)
        ax2.set_title(
            f"Stationary vs Marginal ABV Distribution{title_suffix}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax2.legend(loc="lower right")

        for bar in bars1:
            w = bar.get_width()
            ax2.text(
                w + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%",
                va="center",
                ha="left",
                fontsize=font_size,
                color="#2b4570",
            )

        for bar in bars2:
            w = bar.get_width()
            ax2.text(
                w + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%",
                va="center",
                ha="left",
                fontsize=font_size,
                color="#a04818",
            )
        ax2.set_xlim(0, max(np.max(pi_marg), np.max(pi_stat)) * 100 + 4)

    plt.tight_layout()


def main(
    out_dir: Optional[Union[Path, str]] = (Path(__file__).resolve().parent / "out"),
) -> None:
    out_dir_path = Path(out_dir) if out_dir is not None else None
    checkins = untappd.load_latest_checkins()
    sessions = segment_sessions(checkins)

    # Coarse
    labels_c, p_c, p_rel_c, stat_c, marg_c, ratio_c = compute_markov(
        sessions, COARSE_BINS
    )
    plot_markov_abv(
        labels_c,
        p_c,
        stat_c,
        marg_c,
        relative=False,
        title_suffix=" (Coarse)",
        out_file=out_dir_path / "markov_abv_coarse.png" if out_dir_path else None,
    )
    plot_markov_abv(
        labels_c,
        p_rel_c,
        stat_c,
        marg_c,
        pi_ratio=ratio_c,
        relative=True,
        title_suffix=" (Coarse Relative)",
        out_file=(
            out_dir_path / "markov_abv_relative_coarse.png" if out_dir_path else None
        ),
    )

    # Fine
    labels_f, p_f, p_rel_f, stat_f, marg_f, ratio_f = compute_markov(
        sessions, FINE_BINS
    )
    plot_markov_abv(
        labels_f,
        p_f,
        stat_f,
        marg_f,
        relative=False,
        title_suffix=" (Fine)",
        out_file=out_dir_path / "markov_abv_fine.png" if out_dir_path else None,
    )
    plot_markov_abv(
        labels_f,
        p_rel_f,
        stat_f,
        marg_f,
        pi_ratio=ratio_f,
        relative=True,
        title_suffix=" (Fine Relative)",
        out_file=(
            out_dir_path / "markov_abv_relative_fine.png" if out_dir_path else None
        ),
    )


if __name__ == "__main__":
    main()
