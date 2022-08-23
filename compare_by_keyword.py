from collections import Counter

import matplotlib.pyplot as plt

import untappd
import untappd_utils


CHECKINS = untappd.load_latest_checkins()

FRUIT = [
    "cherry",
    "blueberry",
    "mango",
    "apple",
    "banana",
    "passion",
    # "guava",
]
HOPS = [
    "chinook",
    "mosaic",
    "simcoe",
    "cascade",
    "citra",
]
    

XS = [i / 4 for i in range(1, 21)]

for keyword in FRUIT:
    # look for anywhere in checkin; probably simplest
    # (could just look in beer name and comment)
    counts = Counter(c.rating for c in CHECKINS if keyword in str(c).lower())
    plt.plot(
        *untappd_utils.smooth_ratings(XS, [counts.get(x, 0) for x in XS]),
        label=keyword,
    )

plt.legend()
plt.show()
