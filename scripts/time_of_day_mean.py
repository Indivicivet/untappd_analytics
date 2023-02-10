from collections import defaultdict

import matplotlib.pyplot as plt

import untappd
import untappd_utils


@untappd_utils.show_or_save_to_out_file
def show_average_rating_by_time(data):
    category_data = defaultdict(lambda: defaultdict(int))
    for checkin in data:
        category_data[checkin.datetime.hour][checkin.rating or 0] += 1

    fig, ax1 = plt.subplots(figsize=(12.8, 7.2))
    ax2 = ax1.twinx()
    counts = {
        cat: sum(rating_counts.values())
        for cat, rating_counts in category_data.items()
    }
    y_data = {
        cat: (
            sum(rating * count for rating, count in rating_counts.items())
            / counts[cat]
        )
        for cat, rating_counts in category_data.items()
    }
    x_data = range(24)
    ax1.plot(x_data, [y_data.get(x, 0) for x in x_data], label="average rating")
    ax1.set_ylabel("rating")
    ax1.set_xlabel("time of day")
    ax2.plot(x_data, [counts.get(x, 0) for x in x_data], label="checkins", color="orange")
    ax2.set_ylabel("checkins")
    fig.legend()


DATA = untappd.load_latest_checkins()
show_average_rating_by_time(DATA)
