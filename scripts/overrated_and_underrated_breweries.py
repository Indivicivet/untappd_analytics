from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

import untappd

CHECKINS = untappd.load_latest_checkins()

brewery_checkins = defaultdict(list)

for c in CHECKINS:
    brewery_checkins[c.beer.brewery].append(c)

SHOW_N = 25
MIN_CHECKINS = 10
USE_EMPIRICAL_BAYES = False  # dominated by regression to the mean?

scores_personal_global = [
    (
        untappd.magic_rating(checkins, average_score_weight=1)[0],
        global_rating,
        brewery,
        len(checkins),
    )
    for brewery, checkins in brewery_checkins.items()
    if (
        global_rating := untappd.magic_rating(
            checkins, use_global=True, average_score_weight=1
        )[0]
    )
    != 0
    and (
        len(checkins) > MIN_CHECKINS if not USE_EMPIRICAL_BAYES else len(checkins) >= 3
    )
]

if USE_EMPIRICAL_BAYES:
    # slop :)
    raw_deltas = [my_r - glob_r for my_r, glob_r, _, _ in scores_personal_global]
    counts = [n for _, _, _, n in scores_personal_global]
    mu_delta = float(np.mean(raw_deltas))
    var_raw = float(np.var(raw_deltas, ddof=1))

    # Estimate typical within-brewery variance
    sigma2_within = 0.35**2  # typical single check-in residual variance
    avg_n = float(np.mean(counts))
    tau2 = max(1e-4, var_raw - sigma2_within / avg_n)

    # Compute shrunken deltas and weights
    eb_records = []
    for my_r, glob_r, brewery, n in scores_personal_global:
        raw_delta = my_r - glob_r
        w = tau2 / (tau2 + sigma2_within / n)
        shrunken_delta = w * raw_delta + (1 - w) * mu_delta
        eb_records.append((my_r, glob_r, brewery, raw_delta, shrunken_delta, w, n))

    print(
        f"using Empirical Bayes shrinkage (prior mean bias={mu_delta:.2f}, tau={np.sqrt(tau2):.2f})"
    )
    print(f"evaluating {len(eb_records)}-many breweries (min 3 checkins)")
    print()

    scores_sorted = sorted(eb_records, key=lambda t: t[4])
    print("most overrated breweries (in my opinion, EB shrunken):")
    for my_r, glob_r, brewery, raw_d, shrunken_d, w, n in scores_sorted[:SHOW_N]:
        print(
            f"{brewery} (N={n}) || my rating: {my_r:.2f}, global: {glob_r:.2f}"
            f" | shrunken delta: {-shrunken_d:.2f} (raw: {-raw_d:.2f}, shrinkage: {1-w:.0%})"
        )

    print()
    print("most underrated breweries (in my opinion, EB shrunken):")
    for my_r, glob_r, brewery, raw_d, shrunken_d, w, n in scores_sorted[-SHOW_N:][::-1]:
        print(
            f"{brewery} (N={n}) || my rating: {my_r:.2f}, global: {glob_r:.2f}"
            f" | shrunken delta: {+shrunken_d:.2f} (raw: {+raw_d:.2f}, shrinkage: {1-w:.0%})"
        )
else:
    print(f"using {MIN_CHECKINS=}")
    print(f"that leaves {len(scores_personal_global)}-many breweries")
    print()

    scores_sorted = sorted(scores_personal_global, key=lambda t: t[0] - t[1])
    print("most overrated breweries (in my opinion):")
    for my_rating, global_rating, brewery, _ in scores_sorted[:SHOW_N]:
        print(
            f"{brewery} || my rating: {my_rating:.2f}"
            f", global_rating: {global_rating:.2f}"
            f", delta = {global_rating - my_rating:.2f}"
        )

    print()
    print("most underrated breweries (in my opinion):")
    for my_rating, global_rating, brewery, _ in scores_sorted[-SHOW_N:][::-1]:
        print(
            f"{brewery} || my rating: {my_rating:.2f}"
            f", global_rating: {global_rating:.2f}"
            f", delta = {my_rating - global_rating:.2f}"
        )
