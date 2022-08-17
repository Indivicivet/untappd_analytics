from collections import defaultdict

import matplotlib.pyplot as plt

import untappd


def show_average_rating_by_time(data, out_file=None):
    category_data = defaultdict(lambda: defaultdict(int))
    for checkin in data:
        category_data[checkin.datetime.hour][checkin.rating or 0] += 1

    x_data = [i / 4 for i in range(1, 21)]
    for label, y_data in sorted(category_data.items()):
        plt.plot(x_data, [y_data.get(x, 0) for x in x_data], label=label)
    plt.legend()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)


DATA = untappd.load_latest_checkins()
show_average_rating_by_time(DATA)
