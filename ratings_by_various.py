from collections import defaultdict

import matplotlib.pyplot as plt

import untappd


def show_histogram(data, func=None, out_file=None):
    if func is None:
        func = lambda checkin: checkin.beer.get_style_category()
    category_data = defaultdict(lambda: defaultdict(int))
    for checkin in data:
        category_data[func(checkin)][checkin.rating or 0] += 1

    x_data = [i / 4 for i in range(1, 21)]
    for label, y_data in category_data.items():
        plt.plot(x_data, [y_data.get(x, 0) for x in x_data], label=label)
    plt.legend()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)


DATA = untappd.load_latest_checkins()
show_histogram(
    DATA,
    func=lambda checkin: checkin.datetime.hour,
)
