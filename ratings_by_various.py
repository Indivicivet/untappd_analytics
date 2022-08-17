from collections import defaultdict

import matplotlib.pyplot as plt

import untappd


def show_category_histogram(data):
    category_data = defaultdict(lambda: defaultdict(int))
    for checkin in data:
        category_data[checkin.beer.get_style_category()][checkin.rating or 0] += 1

    x_data = [i / 4 for i in range(1, 21)]
    for label, y_data in category_data.items():
        plt.plot(x_data, [y_data.get(x, 0) for x in x_data], label=label)
    plt.legend()
    plt.show()


DATA = untappd.load_latest_checkins()
show_category_histogram(DATA)
