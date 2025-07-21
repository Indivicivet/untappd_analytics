"""
[LLM gen...]
Plot probabilities related to subsequent drinking behaviour.

Two plots:
1. P(next drink within 6h | rating = x)
2. P(further drink in session | session average rating = x), where sessions are
   sequences of checkins with <6h gaps between consecutive checkins.
"""

from collections import defaultdict
from datetime import timedelta
from typing import Sequence

import matplotlib.pyplot as plt

import untappd

MAX_GAP_HOURS = 6  # hours separating sessions / defining "shortly after"
RATING_BIN_WIDTH = 0.25  # width of rating bins for smoothing
MIN_COUNT_PER_BIN = 3  # suppress bins with fewer data points


def bin_value(value: float, width: float) -> float:
    """Return the left edge of the bin that ``value`` falls into."""
    return round((value // width) * width, 5)


def probability_next_within_six_hours(
    checkins: Sequence[untappd.Checkin],
) -> tuple[list[float], list[float]]:
    """Compute P(next checkin within 6h | rating bin)."""
    if not checkins:
        return [], []

    # Build list of (rating, has_next_within_window)
    paired: list[tuple[float, bool]] = []
    for i, c in enumerate(checkins):
        if i + 1 >= len(checkins):
            continue  # no next checkin
        next_c = checkins[i + 1]
        time_diff = next_c.datetime - c.datetime  # type: ignore
        within = time_diff <= timedelta(hours=MAX_GAP_HOURS)
        paired.append((c.rating, within))  # type: ignore[arg-type]

    counts: dict[float, list[bool]] = defaultdict(list)
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


# --- Plot 2: Probability a further drink is consumed within a session


def compute_sessions(
    checkins: Sequence[untappd.Checkin],
) -> list[list[untappd.Checkin]]:
    sessions: list[list[untappd.Checkin]] = []
    current: list[untappd.Checkin] = []

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


def probability_further_in_session(
    checkins: Sequence[untappd.Checkin],
) -> tuple[list[float], list[float]]:
    sessions = compute_sessions(checkins)

    # For each checkin except the last in its session, record session average rating
    observations: list[tuple[float, bool]] = []
    for session in sessions:
        if len(session) == 1:
            continue  # singletons provide no positive examples
        session_avg = sum(c.rating for c in session if c.rating is not None) / len(
            session
        )
        for idx, c in enumerate(session):
            has_further = idx < len(session) - 1
            # Skip the last drink (as has_further == False always) if we want a
            # pure conditional? Keep it: it contributes negatives.
            observations.append((session_avg, has_further))

    counts: dict[float, list[bool]] = defaultdict(list)
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


if __name__ == "__main__":  # pragma: no cover
    checkins = untappd.load_latest_checkins()

    xs1, ys1 = probability_next_within_six_hours(checkins)
    plt.figure()
    plt.scatter(xs1, ys1)
    plt.xlabel("Rating bin centre")
    plt.ylabel("P(next drink within 6h)")
    plt.title("Probability of next drink within 6h vs rating")

    xs2, ys2 = probability_further_in_session(checkins)
    plt.figure()
    plt.scatter(xs2, ys2)
    plt.xlabel("Session average rating bin centre")
    plt.ylabel("P(further drink in session)")
    plt.title("Probability of further drink vs session average rating")

    plt.show()
