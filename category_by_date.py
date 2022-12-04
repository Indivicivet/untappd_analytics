import datetime
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import seaborn

import untappd

CHECKINS = untappd.load_latest_checkins()

GROUP_TIMESPAN = datetime.timedelta(days=31 * 4)
start_date = min(c.datetime for c in CHECKINS)
end_date = max(c.datetime for c in CHECKINS) - GROUP_TIMESPAN
day_starts = [
    datetime.datetime.fromordinal(n)
    for n in range(start_date.toordinal(), end_date.toordinal())
]

category_counts = []
for start in day_starts:
    counts_this_time = Counter()
    for ci in CHECKINS:
        if start <= ci.datetime < start + GROUP_TIMESPAN:
            counts_this_time[ci.beer.get_style_category()] += 1
    category_counts.append(counts_this_time)

seaborn.set()

plt.figure(figsize=(12.8, 7.2))
for category in untappd.CATEGORY_KEYWORDS:
    plt.plot(
        day_starts,
        [counts[category] for counts in category_counts],
        label=category,
    )
plt.xlabel("date")
plt.ylabel("checkins")
plt.legend()
plt.show()