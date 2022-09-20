import matplotlib.pyplot as plt
import seaborn

import untappd

CHECKINS = untappd.load_latest_checkins()

times = [checkin.datetime for checkin in CHECKINS]
proportion_uniques = []
n_unique = 0
unique_beers = set()
for i, checkin in enumerate(CHECKINS):
    if checkin.beer not in unique_beers:
        unique_beers.add(checkin.beer)
        n_unique += 1
    proportion_uniques.append(n_unique / (i + 1))

RECENCY_SIZE = 200
recent_uniques = [
    # ew :)
    (prop_b * (i + RECENCY_SIZE + 1) - prop_a * (i + 1)) / RECENCY_SIZE
    for i, (prop_a, prop_b) in enumerate(zip(
        proportion_uniques,
        proportion_uniques[RECENCY_SIZE:],
    ))
]

seaborn.set()

plt.figure(figsize=(12.8, 7.2))
plt.plot(times, proportion_uniques, label="cumulative")
plt.plot(
    times[RECENCY_SIZE//2:-RECENCY_SIZE//2],  # must be even ^^;
    recent_uniques,
    label=f"moving average {RECENCY_SIZE}",
)
plt.ylim(0, 1)
plt.xlabel("date")
plt.ylabel("unique ratio")
plt.legend()
plt.show()
