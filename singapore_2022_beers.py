from collections import Counter

import matplotlib.pyplot as plt
from datetime import datetime

import untappd


CHECKINS = untappd.load_latest_checkins()


def first_after(*datetime_args):
    return next(
        i
        for i, c in enumerate(CHECKINS)
        if c.datetime >= datetime(*datetime_args)
    )


checkins_sg = CHECKINS[
    first_after(2022, 6, 14, 20)
    :first_after(2022, 7, 7)
]
checkins_pre_sg = CHECKINS[
    :first_after(2022, 6, 14, 20)
]

print("Checkins pre-Singapore:", len(checkins_pre_sg))
print("Checkins in Singapore:", len(checkins_sg))


def plot_normalized(checkins):
    hist = Counter(c.rating for c in checkins)
    xs = [x / 4 for x in range(1, 21)]
    plt.plot(
        xs,
        [100 * hist.get(x, 0) / hist.total() for x in xs],
    )


plot_normalized(checkins_sg)
plot_normalized(checkins_pre_sg)
plt.legend(["Singapore beers", "Lifetime beers"])
plt.xlabel("Rating")
plt.ylabel("% of checkins")
plt.show()
