"""
initially a copy-paste of top_beers_normalized_by_style_category.py
"""

# todo :: combine with ratings_vs_abv_stats?
# todo :: consolidate with top_checkins_normalized_by_time_window.py
# and top_beers_normalized_by_style category
# (although need to think about that being checkins, and this being beers...)
# todo :: figure out better normalization (by multiple params?)

import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import untappd
import untappd_utils

CHECKINS = untappd.load_latest_checkins()


# LOWER_BOUNDS + rounding copied from rating_vs_abv_stats.py;
# todo :: think how to consolidate
UPPER_BOUNDS = np.linspace(0, 14.5, 21)


def round_abv(abv):
    return next((x for x in UPPER_BOUNDS if x >= abv), UPPER_BOUNDS[-1])


all_mean, all_std = untappd_utils.mean_and_std(CHECKINS)

# calculate statistics to normalize against
# note: compared to normalized by time, here we go by beer (not by checkin)
# making it different just for fun :) (and here I think it's slightly better)
ratings_by_beer = untappd.average_rating_by_beer(CHECKINS)
ratings_by_abv = {x: [] for x in UPPER_BOUNDS}
for beer, rating in ratings_by_beer.items():
    ratings_by_abv[round_abv(beer.abv)].append(rating)

stats_by_abv = {
    abv: (statistics.mean(ratings), statistics.stdev(ratings))
    for abv, ratings in ratings_by_abv.items()
}


for beer, rating in ratings_by_beer.items():
    abv_mean, abv_std = stats_by_abv[round_abv(beer.abv)]
    beer._normalized_rating = all_mean + all_std * (rating - abv_mean) / abv_std


@untappd_utils.show_or_save_to_out_file
def plot_abv_normalization():
    x_plot_args = {
        "x": UPPER_BOUNDS - UPPER_BOUNDS[1],
        "align": "edge",
        "width": UPPER_BOUNDS[1],
    }
    plt.bar(
        **x_plot_args,
        height=[stats_by_abv[bucket][0] for bucket in UPPER_BOUNDS],
        label="mean",
    )
    plt.bar(
        **x_plot_args,
        height=[stats_by_abv[bucket][1] for bucket in UPPER_BOUNDS],
        label="std",
    )
    plt.bar(
        **x_plot_args,
        height=[
            len(ratings_by_abv[bucket]) * (5 / len(CHECKINS)) for bucket in UPPER_BOUNDS
        ],
        label="number of ratings, arbitrarily scaled",
        alpha=0.5,
    )
    plt.xlabel("abv")
    plt.legend()


if __name__ == "__main__":
    plot_abv_normalization()
