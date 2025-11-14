import datetime
import statistics
from collections import Counter, defaultdict

from matplotlib import pyplot as plt

import untappd


def _get_colour(country_name):
    # Asia
    if "China" in country_name:
        return "red"
    if "Japan" in country_name:
        return "#DDDDDD"
    # NA: blue
    if "United States" in country_name:
        return "darkblue"
    if "Canada" in country_name:
        return "#3333BB"
    return "gray"


def country_pie_and_ratings(cis):
    ratings_by_country = defaultdict(list)
    for ci in cis:
        ratings_by_country[ci.beer.brewery.country].append(ci.rating)
    country_freq = Counter(
        ci.beer.brewery.country
        for ci in cis
    )
    print(country_freq.most_common())

    country_order = [c for c, _ in country_freq.most_common()][::-1]  # top to bottom
    country_cols = [_get_colour(c) for c in country_order]

    fig, (ax_pie, ax_violin) = plt.subplots(1, 2, figsize=(12, 7.2))

    ax_pie.pie(
        [country_freq[c] for c in country_order],
        labels=country_order,
        startangle=90,
        colors=country_cols,
    )

    # taken from rating_histogram_by_various.show_violin()
    violin = ax_violin.violinplot(
        list([ratings_by_country[country] for country in country_order]),
        quantiles=[[0, 0.1, 0.5, 0.9, 1]] * len(country_order),
        vert=False,
        widths=0.75,
        bw_method=0.25 + 0.1,  # known discrete sampling, smoothed a bit extra
    )
    for i, body in enumerate(violin["bodies"]):
        body.set_facecolor(country_cols[i])
        body.set_edgecolor("black")
        body.set_alpha(1)

    ax_violin.set_yticks(
        list(range(1, len(country_order) + 1)),
        labels=[
            f"{c} ({statistics.mean(ratings_by_country[c]):.2f})" for c in country_order
        ],
    )
    plt.show()


if __name__ == "__main__":
    FEST_TAG = "beer celebration"
    ALL_CIS = untappd.load_latest_checkins()
    FEST_CIS = [
        ci
        for ci in ALL_CIS
        if FEST_TAG in ((ci.venue and ci.venue.name) or "").lower()
    ]
    country_pie_and_ratings(FEST_CIS)
    country_pie_and_ratings(
        [
            ci
            for ci in ALL_CIS
            if datetime.datetime(2025, 10, 23) < ci.datetime < datetime.datetime(2025, 11, 9, 23)
        ]
    )
