from collections import defaultdict

import matplotlib.pyplot as plt

import untappd


def show_average_rating_by_time(data, out_file=None):
    category_data = defaultdict(lambda: defaultdict(int))
    for checkin in data:
        category_data[checkin.datetime.hour][checkin.rating or 0] += 1

    y_data = {
        cat: (
            sum(rating * count for rating, count in rating_counts.items())
            / sum(rating_counts.values())
        )
        for cat, rating_counts in category_data.items()
    }
    x_data = range(24)
    plt.plot(x_data, [y_data.get(x, 0) for x in x_data])
    #plt.legend()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)


DATA = untappd.load_latest_checkins()
show_average_rating_by_time(DATA)
