from collections import Counter

import seaborn
from matplotlib import pyplot as plt

import untappd

CIS = untappd.load_latest_checkins()

freqs = Counter()

for c0, c1 in zip(CIS, CIS[1:]):
    freqs[(c0.rating, c1.rating)] += 1

ratings = [x / 4 for x in range(1, 21)]

plt.figure(figsize=(12.8, 7.2))
seaborn.heatmap(
    [[freqs.get((i, j), 0) for j in ratings] for i in ratings],
    xticklabels=ratings,
    yticklabels=ratings,
    annot=True,
    fmt="d",
).invert_yaxis()
plt.xlabel("first checkin")
plt.ylabel("second checkin")
plt.show()
