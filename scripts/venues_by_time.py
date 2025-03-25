from collections import defaultdict, Counter

from matplotlib import pyplot as plt

import untappd

CIS = untappd.load_latest_checkins()
DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

day_counts = defaultdict(lambda: Counter())

for ci in CIS:
    day_counts[ci.datetime.strftime("%A")][ci.venue] += 1

ROWS = 2
COLS = 4
_, axes = plt.subplots(ROWS, COLS, figsize=(12.8, 7.2))
for i, day in enumerate(DAYS):
    top_counts = day_counts[day].most_common(10)
    ax = axes[i // COLS, i % COLS]
    ax.pie(
        [c for _, c in top_counts],
        labels=[v if v is not None else "None" for v, _ in top_counts],
    )
    ax.set_title(day)
    # ax.legend()
plt.show()
