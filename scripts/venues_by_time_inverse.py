# based on venues_by_time but plotting the opposite way around

import datetime
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

venue_counts = defaultdict(lambda: Counter())
venue_total_counts = Counter()

for ci in CIS:
    venue_counts[ci.venue][
        (ci.datetime - datetime.timedelta(hours=5)).strftime("%A")
    ] += 1
    venue_total_counts[ci.venue] += 1


ROWS = 2
COLS = 4
_, axes = plt.subplots(ROWS, COLS, figsize=(12.8, 7.2))
for i, (venue, _) in enumerate(venue_total_counts.most_common(ROWS * COLS)):
    count_dict = venue_counts[venue]
    print(count_dict)
    ax = axes[i // COLS, i % COLS]
    ax.pie(
        [count_dict[day] for day in DAYS],
        labels=[f"{day} ({count_dict[day]})" for day in DAYS],
    )
    ax.set_title(venue)
    # ax.legend()
plt.show()
