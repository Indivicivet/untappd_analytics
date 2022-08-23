from collections import defaultdict
import datetime

import matplotlib.pyplot as plt

import untappd
import untappd_utils


class SessionTracker:
    def __init__(self, cap=5):
        self.session = [untappd.Checkin(beer=None, datetime=datetime.datetime.min)]
        self.cap = cap

    def session_n(self, checkin):
        if checkin.datetime > self.session[-1].datetime + datetime.timedelta(hours=1):
            self.session = []
        self.session.append(checkin)
        n = len(self.session)
        if n >= self.cap:
            return f"{self.cap}+"
        return str(n)


def show_histogram(data, func, out_file=None):
    category_data = defaultdict(lambda: defaultdict(int))
    for checkin in data:
        category_data[func(checkin)][checkin.rating or 0] += 1

    x_data = [i / 4 for i in range(1, 21)]
    for label, counts in sorted(category_data.items()):
        y_data = [counts.get(x, 0) for x in x_data]
        plt.plot(*untappd_utils.smooth_ratings(x_data, y_data), label=label)
    plt.legend()
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)


DATA = untappd.load_latest_checkins()
show_histogram(
    DATA,
    # func=lambda checkin: checkin.beer.get_style_category(),
    # func=lambda checkin: checkin.datetime.hour,
    # func=SessionTracker().session_n,
    func=(lambda checkin:
        0.5 if checkin.beer.abv < 5
        else 5.75 if checkin.beer.abv < 7.5
        else 7.510 if checkin.beer.abv < 10
        else 10.999
    ),
)
