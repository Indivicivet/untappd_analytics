import matplotlib.pyplot as plt

import untappd

CHECKINS = untappd.load_latest_checkins()

MIN_ABV = 12

cis_high_abv = [c for c in CHECKINS if c.beer.abv > MIN_ABV]

plt.figure(figsize=(12.8, 7.2))
plt.scatter(
    [c.beer.abv for c in cis_high_abv],
    [c.rating for c in cis_high_abv],
    alpha=0.3,
    s=200,
)
plt.xlabel("abv")
plt.ylabel("rating")
plt.show()
