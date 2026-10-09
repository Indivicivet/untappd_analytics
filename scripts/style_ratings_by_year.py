import statistics

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn

import untappd
import untappd_utils


@untappd_utils.show_or_save_to_out_file
def plot_style_ratings_by_year(checkins):
    start = min(c.datetime for c in checkins)
    end = max(c.datetime for c in checkins)
    by_year = {
        cat: {
            year: statistics.mean(
                [
                    c.rating
                    for c in checkins
                    if c.datetime.year == year and c.beer.get_style_category() == cat
                ]
            )
            for year in range(start.year, end.year + 1)
        }
        for cat in untappd.CATEGORY_KEYWORDS
    }

    seaborn.set()
    plt.figure(figsize=(12.8, 7.2))
    for year, checkins_by_cat in by_year.items():
        plt.bar(
            [f"{year} {cat}" for cat in checkins_by_cat.keys()],
            checkins_by_cat.values(),
        )
    plt.xticks(rotation=-45)

    all_vals = [v for year in by_year.values() for v in year.values()]
    plt.ylim([min(all_vals) - 0.25, max(all_vals)])
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.ylabel("average rating")


if __name__ == "__main__":
    CHECKINS = untappd.load_latest_checkins()
    plot_style_ratings_by_year(CHECKINS)
