import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn

import untappd

CHECKINS = untappd.load_latest_checkins()

MIN_ABV = 12

cis_high_abv = [c for c in CHECKINS if c.beer.abv >= MIN_ABV]

seaborn.set()
plt.figure(figsize=(12.8, 7.2))
plt.scatter(
    [c.beer.abv for c in cis_high_abv],
    [c.rating for c in cis_high_abv],
    alpha=0.3,
    s=200,
)
ax = plt.gca()
ax.set_xscale("log")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
plt.xlabel("abv")
plt.ylabel("rating")
plt.show()
