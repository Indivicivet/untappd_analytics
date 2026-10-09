import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn

import untappd
import untappd_utils

MIN_ABV = 12


@untappd_utils.show_or_save_to_out_file
def plot_high_abv_ratings(checkins):
    cis_high_abv = [c for c in checkins if c.beer.abv >= MIN_ABV]

    seaborn.set()
    plt.figure(figsize=(12.8, 7.2))
    plt.scatter(
        [c.beer.abv for c in cis_high_abv],
        [c.rating for c in cis_high_abv],
        alpha=0.3,
        s=200,
    )
    ax = plt.gca()
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel("abv")
    plt.ylabel("rating")


if __name__ == "__main__":
    CHECKINS = untappd.load_latest_checkins()
    plot_high_abv_ratings(CHECKINS)
