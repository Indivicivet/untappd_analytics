import statistics
from collections import defaultdict

import numpy as np
import seaborn
from matplotlib import pyplot as plt

import untappd
import untappd_utils

UPPER_BOUNDS = np.linspace(0, 14.5, 21)


def plot_rating_vs_abv(checkins, tag="", unify_color=False):
    ratings_by_bucket = {x: [] for x in UPPER_BOUNDS}
    for ci in checkins:
        upper_bound = next(
            (x for x in UPPER_BOUNDS if x >= ci.beer.abv),
            UPPER_BOUNDS[-1],
        )
        ratings_by_bucket[upper_bound].append(ci)

    present_upper_bounds, *things_to_plot = list(zip(*[
        (bound, *untappd_utils.mean_plus_minus_std(cis))
        for bound, cis in ratings_by_bucket.items()
        if len(cis) >= 2
    ]))

    slope, offset = np.polyfit(
        present_upper_bounds,
        things_to_plot[1],
        1,
        w=[
            1 if 4 <= x <= 12 else 0.5
            for x in present_upper_bounds
        ],
    )
    best_fit_means = [x * slope + offset for x in UPPER_BOUNDS]

    color = None
    for thing, label, alpha in zip(
        things_to_plot,
        ["mean - 1std", "mean", "mean + 1std"],
        [0.5, 1, 0.5],
    ):
        lines = plt.plot(
            present_upper_bounds,
            thing,
            label=f"{tag}{label}",
            alpha=alpha,
            color=color,
        )
        if unify_color and color is None:
            color = lines[0].get_color()
    plt.plot(
        UPPER_BOUNDS,
        best_fit_means,
        label=f"{tag}weighted linear fit mean {slope=:.3f} {offset=:.3f}",
        linestyle="dashed",
        color=color,
    )


if __name__ == "__main__":
    CHECKINS = untappd.load_latest_checkins()
    # STYLES = None
    # STYLES = ["stout", "sour", "ipa"]
    STYLES = ["stout"]

    seaborn.set()
    plt.figure(figsize=(12.8, 7.2))
    if STYLES is None:
        plot_rating_vs_abv(CHECKINS)
    else:
        for style in STYLES:
            plot_rating_vs_abv(
                [c for c in CHECKINS if c.beer.get_style_category() == style],
                tag=f"[{style}] ",
                unify_color=True,
            )
    plt.legend()
    plt.show()
