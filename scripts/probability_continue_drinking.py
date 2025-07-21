"""
[LLM gen...]
Plot probabilities related to subsequent drinking behaviour.

Two plots:
1. P(next drink within 6h | rating = x)
2. P(further drink in session | session average rating = x), where sessions are
   sequences of checkins with <6h gaps between consecutive checkins.

Usage: place this script alongside ``untappd.py`` (the library provided) and run.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

import untappd

# --- Configuration -----------------------------------------------------------------
MAX_GAP_HOURS = 6  # hours separating sessions / defining "shortly after"
RATING_BIN_WIDTH = 0.25  # width of rating bins for smoothing
MIN_COUNT_PER_BIN = 3  # suppress bins with fewer data points

# ------------------------------------------------------------------------------------


def load_checkins() -> List[untappd.Checkin]:
    checkins = untappd.load_latest_checkins(ignore_unrated=True, ignore_tasters=False)
    # Ensure chronological order
    checkins.sort(key=lambda c: c.datetime or 0)
    # Filter out any without timestamps or ratings just in case
    return [c for c in checkins if c.datetime and c.rating is not None]


def bin_value(value: float, width: float) -> float:
    """Return the left edge of the bin that ``value`` falls into."""
    return round((value // width) * width, 5)


# --- Plot 1: Probability of a drink within 6 hours after a checkin ------------------

def probability_next_within_six_hours(checkins: Sequence[untappd.Checkin]) -> Tuple[List[float], List[float]]:
    """Compute P(next checkin within 6h | rating bin)."""
    if not checkins:
        return [], []

    # Build list of (rating, has_next_within_window)
    paired: List[Tuple[float, bool]] = []
    for i, c in enumerate(checkins):
        if i + 1 >= len(checkins):
            continue  # no next checkin
        next_c = checkins[i + 1]
        time_diff = next_c.datetime - c.datetime  # type: ignore
        within = time_diff <= timedelta(hours=MAX_GAP_HOURS)
        paired.append((c.rating, within))  # type: ignore[arg-type]

    counts: dict[float, List[bool]] = defaultdict(list)
    for rating, within in paired:
        counts[bin_value(rating, RATING_BIN_WIDTH)].append(within)

    xs, ys = [], []
    for b in sorted(counts):
        vals = counts[b]
        if len(vals) < MIN_COUNT_PER_BIN:
            continue
        xs.append(b + RATING_BIN_WIDTH / 2)  # centre of bin
        ys.append(sum(vals) / len(vals))
    return xs, ys


# --- Plot 2: Probability a further drink is consumed within a session ----------------

def compute_sessions(checkins: Sequence[untappd.Checkin]) -> List[List[untappd.Checkin]]:
    sessions: List[List[untappd.Checkin]] = []
    current: List[untappd.Checkin] = []

    for c in checkins:
        if not current:
            current.append(c)
            continue
        prev = current[-1]
        gap = c.datetime - prev.datetime  # type: ignore
        if gap <= timedelta(hours=MAX_GAP_HOURS):
            current.append(c)
        else:
            sessions.append(current)
            current = [c]
    if current:
        sessions.append(current)
    return sessions


def probability_further_in_session(checkins: Sequence[untappd.Checkin]) -> Tuple[List[float], List[float]]:
    sessions = compute_sessions(checkins)

    # For each checkin except the last in its session, record session average rating
    observations: List[Tuple[float, bool]] = []
    for session in sessions:
        if len(session) == 1:
            continue  # singletons provide no positive examples
        session_avg = sum(c.rating for c in session if c.rating is not None) / len(session)  # type: ignore[arg-type]
        for idx, c in enumerate(session):
            has_further = idx < len(session) - 1
            # Skip the last drink (as has_further == False always) if we want a pure conditional? Keep it: it contributes negatives.
            observations.append((session_avg, has_further))

    counts: dict[float, List[bool]] = defaultdict(list)
    for avg_rating, has_further in observations:
        counts[bin_value(avg_rating, RATING_BIN_WIDTH)].append(has_further)

    xs, ys = [], []
    for b in sorted(counts):
        vals = counts[b]
        if len(vals) < MIN_COUNT_PER_BIN:
            continue
        xs.append(b + RATING_BIN_WIDTH / 2)
        ys.append(sum(vals) / len(vals))
    return xs, ys


# --- Main ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    checkins = load_checkins()
    if not checkins:
        print("No checkins loaded.")
        return

    # Plot 1
    xs1, ys1 = probability_next_within_six_hours(checkins)
    plt.figure()
    plt.scatter(xs1, ys1)
    plt.xlabel("Rating bin centre")
    plt.ylabel("P(next drink within 6h)")
    plt.title("Probability of next drink within 6h vs rating")

    # Plot 2
    xs2, ys2 = probability_further_in_session(checkins)
    plt.figure()
    plt.scatter(xs2, ys2)
    plt.xlabel("Session average rating bin centre")
    plt.ylabel("P(further drink in session)")
    plt.title("Probability of further drink vs session average rating")

    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
