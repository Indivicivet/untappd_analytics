import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn

import untappd
import untappd_utils

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MAX_SESSION_GAP_HOURS = 6
HEAVY_STYLES = {"Imperial Stout", "Imperial / DIPA"}

STYLE_FAMILIES = [
    ("Hoppy Ales", ["Pale / Session Ale", "IPA", "Imperial / DIPA"]),
    ("Dark & Malty", ["Stout / Porter", "Imperial Stout", "Barleywine"]),
    ("Sour & Belgian", ["Sour / Wild Ale", "Belgian / Farmhouse"]),
    ("Crisp & Refreshing", ["Wheat Beer", "Lager / Pilsner"]),
    ("Other", ["Other Ale", "Cider / Mead", "Other"]),
]
DEFAULT_STYLE_ORDER = [s for _, group in STYLE_FAMILIES for s in group]


def categorize_style(checkin: untappd.Checkin) -> str:
    """Categorize a checkin into a granular beer style."""
    beer_type = checkin.beer.type.lower()
    abv = checkin.beer.abv if checkin.beer.abv is not None else 0.0

    if (
        "barleywine" in beer_type
        or "barley wine" in beer_type
        or "wheat wine" in beer_type
    ):
        return "Barleywine"
    if (
        ("imperial" in beer_type and "stout" in beer_type)
        or "pastry stout" in beer_type
        or (("stout" in beer_type or "porter" in beer_type) and abv >= 8.5)
    ):
        return "Imperial Stout"
    if "stout" in beer_type or "porter" in beer_type:
        return "Stout / Porter"
    if (
        (
            "double" in beer_type
            or "triple" in beer_type
            or "quad" in beer_type
            or "imperial" in beer_type
        )
        and "ipa" in beer_type
    ) or ("ipa" in beer_type and abv >= 8.0):
        return "Imperial / DIPA"
    if "ipa" in beer_type:
        return "IPA"
    if "pale ale" in beer_type or "bitter" in beer_type or "session" in beer_type:
        return "Pale / Session Ale"
    if (
        "sour" in beer_type
        or "lambic" in beer_type
        or "wild ale" in beer_type
        or "gose" in beer_type
    ):
        return "Sour / Wild Ale"
    if "lager" in beer_type or "pilsner" in beer_type or "helles" in beer_type:
        return "Lager / Pilsner"
    if "wheat" in beer_type or "hefeweizen" in beer_type or "witbier" in beer_type:
        return "Wheat Beer"
    if (
        "belgian" in beer_type
        or "tripel" in beer_type
        or "dubbel" in beer_type
        or "quadrupel" in beer_type
        or "saison" in beer_type
    ):
        return "Belgian / Farmhouse"
    if "cider" in beer_type or "mead" in beer_type:
        return "Cider / Mead"

    category = checkin.beer.get_style_category()
    if category == "stout":
        return "Stout / Porter"
    if category == "sour":
        return "Sour / Wild Ale"
    if category == "ipa":
        return "IPA"
    if category == "lager":
        return "Lager / Pilsner"
    if category == "wheat":
        return "Wheat Beer"
    if category == "other ale":
        return "Other Ale"
    return "Other"


def segment_sessions(
    checkins: Sequence[untappd.Checkin],
    max_gap: timedelta = timedelta(hours=MAX_SESSION_GAP_HOURS),
) -> list[list[untappd.Checkin]]:
    """Segment checkins into drinking sessions based on maximum time gap."""
    sessions: list[list[untappd.Checkin]] = []
    current_session: list[untappd.Checkin] = []

    for c in checkins:
        if not current_session:
            current_session.append(c)
            continue
        prev = current_session[-1]
        if (
            c.datetime is not None
            and prev.datetime is not None
            and (c.datetime - prev.datetime) <= max_gap
        ):
            current_session.append(c)
        else:
            sessions.append(current_session)
            current_session = [c]
    if current_session:
        sessions.append(current_session)
    return sessions


def extract_transitions(
    sessions: Sequence[Sequence[untappd.Checkin]],
    style_fn: Callable[[untappd.Checkin], str] = categorize_style,
) -> tuple[dict[str, Counter], Counter, Counter]:
    """Extract consecutive and terminal style transitions across sessions."""
    transition_counts: dict[str, Counter] = defaultdict(Counter)
    start_counts: Counter = Counter()
    marginal_counts: Counter = Counter()

    for session in sessions:
        styles = [style_fn(c) for c in session]
        start_counts[styles[0]] += 1
        for style in styles:
            marginal_counts[style] += 1
        for i in range(len(styles) - 1):
            transition_counts[styles[i]][styles[i + 1]] += 1
        transition_counts[styles[-1]]["Session End"] += 1

    return transition_counts, start_counts, marginal_counts


def compute_markov_matrices(
    transition_counts: dict[str, Counter],
    marginal_counts: Counter,
    styles_order: Optional[Sequence[str]] = None,
) -> tuple[
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
]:
    """Construct transition matrices, stationary vs marginal distributions, and family divider indices."""
    base_order = styles_order if styles_order is not None else DEFAULT_STYLE_ORDER
    styles_list = [s for s in base_order if s in marginal_counts]
    for s, _ in marginal_counts.most_common():
        if s not in styles_list:
            styles_list.append(s)

    col_labels = styles_list + ["Session End"]

    # Compute family dividers for visual grouping
    family_dividers: list[int] = []
    curr_idx = 0
    for _, group in STYLE_FAMILIES:
        group_present = [s for s in group if s in styles_list]
        curr_idx += len(group_present)
        if 0 < curr_idx < len(styles_list):
            family_dividers.append(curr_idx)

    # Full transition matrix: rows = current styles, cols = next styles + Session End
    p_full = np.zeros((len(styles_list), len(col_labels)))
    for i, s1 in enumerate(styles_list):
        total = sum(transition_counts[s1].values())
        for j, s2 in enumerate(styles_list):
            p_full[i, j] = transition_counts[s1][s2] / total if total > 0 else 0
        p_full[i, len(styles_list)] = (
            transition_counts[s1]["Session End"] / total if total > 0 else 0
        )

    # Style-to-style matrix (conditioned on continuing within session)
    p_cont = np.zeros((len(styles_list), len(styles_list)))
    for i, s1 in enumerate(styles_list):
        total_cont = sum(transition_counts[s1][s2] for s2 in styles_list)
        for j, s2 in enumerate(styles_list):
            p_cont[i, j] = (
                transition_counts[s1][s2] / total_cont if total_cont > 0 else 0
            )

    # Stationary distribution pi P_cont = pi
    evals, evecs = np.linalg.eig(p_cont.T)
    idx = int(np.argmin(np.abs(evals - 1.0)))
    pi_stat = np.real(evecs[:, idx])
    pi_stat = pi_stat / np.sum(pi_stat)

    total_marginal = sum(marginal_counts.values())
    pi_marg = np.array([marginal_counts[s] / total_marginal for s in styles_list])

    return (
        styles_list,
        col_labels,
        p_full,
        p_cont,
        pi_stat,
        pi_marg,
        family_dividers,
    )


def analyze_terminal_states(
    styles_list: list[str],
    transition_counts: dict[str, Counter],
) -> list[tuple[str, float, int, int]]:
    """Analyze which styles act as terminal states (ending session)."""
    terminal_stats = []
    for s in styles_list:
        total = sum(transition_counts[s].values())
        p_end = transition_counts[s]["Session End"] / total if total > 0 else 0
        terminal_stats.append((s, p_end, transition_counts[s]["Session End"], total))
    terminal_stats.sort(key=lambda item: item[1], reverse=True)
    return terminal_stats


def analyze_palate_cleansers(
    sessions: Sequence[Sequence[untappd.Checkin]],
    style_fn: Callable[[untappd.Checkin], str] = categorize_style,
) -> tuple[Counter, Counter, int]:
    """Analyze whether sours/lighter styles act as palate cleansers between heavy beers."""
    middle_after_heavy: Counter = Counter()
    triplet_counts: Counter = Counter()
    total_triplets = 0

    for session in sessions:
        if len(session) < 3:
            continue
        styles = [style_fn(c) for c in session]
        for i in range(1, len(styles) - 1):
            prev_st, curr_st, next_st = styles[i - 1], styles[i], styles[i + 1]
            if prev_st in HEAVY_STYLES:
                middle_after_heavy[curr_st] += 1
                if next_st in HEAVY_STYLES:
                    triplet_counts[curr_st] += 1
                    total_triplets += 1

    return middle_after_heavy, triplet_counts, total_triplets


@untappd_utils.show_or_save_to_out_file
def plot_markov_style_transitions(
    styles_list: list[str],
    col_labels: list[str],
    p_full: np.ndarray,
    pi_stat: np.ndarray,
    pi_marg: np.ndarray,
    family_dividers: Optional[Sequence[int]] = None,
) -> None:
    """Generate dual-panel visualization: Transition Heatmap and Stationary vs Marginal Bar Chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 9.5))

    # Panel 1: Transition Heatmap
    seaborn.heatmap(
        p_full,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=col_labels,
        yticklabels=styles_list,
        cbar_kws={"label": "Transition Probability P(Next | Current)"},
        ax=ax1,
        linewidths=0.5,
        linecolor="#e0e0e0",
    )

    if family_dividers:
        for div in family_dividers:
            ax1.axhline(div, color="#333333", linewidth=1.5, linestyle="-")
            ax1.axvline(div, color="#333333", linewidth=1.5, linestyle="-")
    ax1.axvline(len(styles_list), color="#d62728", linewidth=1.8, linestyle="--")

    ax1.set_title(
        "First-Order Markov Style Transition Matrix P(Next | Current)\n[Grouped by Beer Style Family]",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xlabel("Next State (Beer Style or Session End)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Current Beer Style", fontsize=11, labelpad=8)
    ax1.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9.5)
    ax1.set_yticklabels(styles_list, rotation=0, fontsize=9.5)

    # Panel 2: Stationary vs Marginal Distribution Bar Chart
    y_pos = np.arange(len(styles_list))
    bar_width = 0.38

    bars1 = ax2.barh(
        y_pos - bar_width / 2,
        pi_marg * 100,
        bar_width,
        label="Marginal Distribution (Overall Check-in %)",
        color="#4C72B0",
        alpha=0.9,
    )
    bars2 = ax2.barh(
        y_pos + bar_width / 2,
        pi_stat * 100,
        bar_width,
        label="Stationary Distribution (Within-Session Steady-State %)",
        color="#DD8452",
        alpha=0.9,
    )

    if family_dividers:
        for div in family_dividers:
            ax2.axhline(div - 0.5, color="#888888", linewidth=1.0, linestyle=":")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(styles_list, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Probability (%)", fontsize=11, labelpad=8)
    ax2.set_title(
        "Markov Stationary vs Marginal Style Distribution\n[Grouped by Beer Style Family]",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax2.legend(loc="lower right", frameon=True, fontsize=10)

    for bar in bars1:
        width = bar.get_width()
        ax2.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#2b4570",
        )

    for bar in bars2:
        width = bar.get_width()
        ax2.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#a04818",
        )

    ax2.set_xlim(0, max(np.max(pi_marg), np.max(pi_stat)) * 100 + 4)
    plt.tight_layout()


def main(
    out_file: Optional[Union[Path, str]] = (
        Path(__file__).resolve().parent / "out" / "markov_style_transitions.png"
    ),
) -> None:
    checkins = untappd.load_latest_checkins()
    sessions = segment_sessions(checkins)

    transition_counts, _, marginal_counts = extract_transitions(sessions)
    (
        styles_list,
        col_labels,
        p_full,
        p_cont,
        pi_stat,
        pi_marg,
        family_dividers,
    ) = compute_markov_matrices(transition_counts, marginal_counts)

    print(
        f"Analyzed {len(checkins)} checkins across {len(sessions)} drinking sessions "
        f"(max gap = {MAX_SESSION_GAP_HOURS}h).\n"
    )

    print("=" * 70)
    print("1. TERMINAL STATE ANALYSIS: P(Session End | Current Style)")
    print("=" * 70)
    terminal_stats = analyze_terminal_states(styles_list, transition_counts)
    for style, p_end, end_cnt, total_cnt in terminal_stats:
        print(
            f"  {style:22s}: P(End) = {p_end:5.1%} ({end_cnt:4d} / {total_cnt:4d} checkins)"
        )

    print("\n" + "=" * 70)
    print("2. PALATE CLEANSER ANALYSIS: Heavy -> Middle -> Heavy Sequences")
    print("   (Heavy Styles: Imperial Stout, Imperial / DIPA)")
    print("=" * 70)
    middle_after_heavy, triplet_counts, total_triplets = analyze_palate_cleansers(
        sessions
    )
    print(f"Total Heavy -> Middle -> Heavy triplets: {total_triplets}")
    print("Middle style breakdown in Heavy -> Middle -> Heavy sequences:")
    for style, cnt in triplet_counts.most_common():
        pct = cnt / total_triplets if total_triplets > 0 else 0
        print(f"  {style:22s}: {cnt:4d} ({pct:5.1%})")

    print("\n" + "=" * 70)
    print("3. MARKOV CHAIN STATIONARY VS MARGINAL STYLE DISTRIBUTION")
    print("=" * 70)
    print(f"{'Style':22s} | {'Marginal':>9s} | {'Stationary':>11s} | {'Delta':>8s}")
    print("-" * 58)
    for i, style in enumerate(styles_list):
        p_m = pi_marg[i]
        p_s = pi_stat[i]
        delta = p_s - p_m
        print(f"{style:22s} | {p_m:8.2%} | {p_s:10.2%} | {delta:+7.2%}")

    print("\nGenerating visualization...")
    plot_markov_style_transitions(
        styles_list,
        col_labels,
        p_full,
        pi_stat,
        pi_marg,
        family_dividers=family_dividers,
        out_file=out_file,
    )


if __name__ == "__main__":
    main()
