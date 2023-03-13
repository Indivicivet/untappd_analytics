import statistics
from collections import defaultdict

import numpy as np
import seaborn
from matplotlib import pyplot as plt

import untappd
import untappd_utils

UPPER_BOUNDS = np.linspace(0, 14.5, 21)


def plot_rating_vs_abv(checkins):
    ratings_by_bucket = {x: [] for x in UPPER_BOUNDS}
    for ci in checkins:
        upper_bound = next(
            (x for x in UPPER_BOUNDS if x >= ci.beer.abv),
            UPPER_BOUNDS[-1],
        )
        ratings_by_bucket[upper_bound].append(ci)

    things_to_plot = list(zip(*[
        untappd_utils.mean_plus_minus_std(cis)
        for cis in ratings_by_bucket.values()
    ]))

    slope, offset = np.polyfit(
        UPPER_BOUNDS,
        things_to_plot[1],
        1,
        w=[
            1 if 4 <= x <= 12 else 0.5
            for x in UPPER_BOUNDS
        ],
    )
    best_fit_means = [x * slope + offset for x in UPPER_BOUNDS]

    for thing, label in zip(things_to_plot, ["mean - 1std", "mean", "mean + 1std"]):
        plt.plot(UPPER_BOUNDS, thing, label=label)
    plt.plot(
        UPPER_BOUNDS,
        best_fit_means,
        label=f"weighted linear fit mean {slope=:.3f} {offset=:.3f}",
    )


if __name__ == "__main__":
    CHECKINS = untappd.load_latest_checkins()

    seaborn.set()
    plt.figure(figsize=(12.8, 7.2))
    plot_rating_vs_abv(CHECKINS)
    plt.legend()
    plt.show()
