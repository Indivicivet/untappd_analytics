from collections import Counter

import matplotlib.pyplot as plt
import seaborn

import untappd
import untappd_utils

seaborn.set()

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
    "centennial",
]
    

XS = [i / 4 for i in range(1, 21)]
scores = []

for keyword in HOPS:
    # look for anywhere in checkin; probably simplest
    # (could just look in beer name and comment)
    counts = Counter(c.rating for c in CHECKINS if keyword in str(c).lower())
    total = sum(counts.values())
    scores.append((
        sum(r * n for r, n in counts.items()) / total,
        keyword,
        total,
    ))
    plt.plot(
        *untappd_utils.smooth_ratings(XS, [counts.get(x, 0) for x in XS]),
        label=keyword,
    )

for rating, keyword, count in sorted(scores, reverse=True):
    print(f"{rating: .2f} {keyword}  ({count})")

plt.legend()
plt.show()
