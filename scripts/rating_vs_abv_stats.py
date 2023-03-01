import statistics
from collections import defaultdict

import numpy as np
import seaborn
from matplotlib import pyplot as plt

import untappd
import untappd_utils

LOWER_BOUNDS = np.linspace(0, 14.5, 21)

CHECKINS = untappd.load_latest_checkins()

ratings_by_bucket = {
    x: []
    for x in LOWER_BOUNDS
}
for ci in CHECKINS:
    try:
        lower_bound = next(x for x in LOWER_BOUNDS if x >= ci.beer.abv)
    except StopIteration:
        lower_bound = LOWER_BOUNDS[-1]
    ratings_by_bucket[lower_bound].append(ci)


things_to_plot = zip(*[
    untappd_utils.mean_plus_minus_std(cis)
    for cis in ratings_by_bucket.values()
])
seaborn.set()
plt.figure(figsize=(12.8, 7.2))
for thing, label in zip(things_to_plot, ["mean - 1std", "mean", "mean + 1std"]):
    plt.plot(ratings_by_bucket.keys(), thing, label=label)
plt.legend()
plt.show()
