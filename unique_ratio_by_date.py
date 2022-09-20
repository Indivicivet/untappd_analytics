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

seaborn.set()

plt.figure(figsize=(12.8, 7.2))
plt.plot(times, proportion_uniques, label="proportion unique")
plt.ylim(0, 1)
plt.xlabel("date")
plt.ylabel("unique ratio")
plt.show()
