from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, ndimage

import untappd

CHECKINS = untappd.load_latest_checkins()

KEYWORDS = [
    "cherry",
    "blueberry",
    "mango",
    "apple",
    "banana",
    "passion",
    # "guava",
]


def smooth_ratings(x0, y0, samples=200):
    x_smooth = np.linspace(min(x0), max(x0), samples, endpoint=True)
    y_linear = interpolate.interp1d(x0, y0)(x_smooth)
    return x_smooth, ndimage.gaussian_filter(y_linear, samples * 0.01)
    

XS = [i / 4 for i in range(1, 21)]

for keyword in KEYWORDS:
    # look for anywhere in checkin; probably simplest
    # (could just look in beer name and comment)
    counts = Counter(c.rating for c in CHECKINS if keyword in str(c).lower())
    plt.plot(
        *smooth_ratings(XS, [counts.get(x, 0) for x in XS]),
        label=keyword,
    )

plt.legend()
plt.show()
