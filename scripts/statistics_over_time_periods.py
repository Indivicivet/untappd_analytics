import datetime
import statistics

import matplotlib.pyplot as plt
import seaborn

import untappd

CHECKINS = untappd.load_latest_checkins()

# todo :: much of this code is copied from category_by_date
# once I have nice smoother plots, ideally move functionality
# into a shared file

SHOW_IBUS = False

GROUP_TIMESPAN = datetime.timedelta(days=31 * 3)
start_date = min(c.datetime for c in CHECKINS)
end_date = max(c.datetime for c in CHECKINS) - GROUP_TIMESPAN
day_starts = [
    datetime.datetime.fromordinal(n)
    for n in range(start_date.toordinal(), end_date.toordinal())
]

abvs = [
    statistics.mean([
        ci.beer.abv
        for ci in CHECKINS
        if start <= ci.datetime < start + GROUP_TIMESPAN
        # todo :: don't do this loop every time :P
    ])
    for start in day_starts
]

seaborn.set()

plt.figure(figsize=(12.8, 7.2))
plt.plot(day_starts, abvs)

if SHOW_IBUS:
    # todo :: should have separate axes + legend
    # (then probably don't need a flag :D)
    ibus = [
        0.1  # hacky :)
        * statistics.mean([
            ci.beer.ibu
            for ci in CHECKINS
            if (
                    start <= ci.datetime < start + GROUP_TIMESPAN
                    and ci.beer.ibu > 0
            )
        ])
        for start in day_starts
    ]
    plt.plot(day_starts, ibus)

plt.xlabel("start date")
plt.ylabel("average abv")
plt.show()
