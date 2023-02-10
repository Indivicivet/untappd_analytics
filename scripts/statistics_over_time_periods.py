import datetime
import statistics
from typing import Callable

import matplotlib.pyplot as plt
import seaborn

import untappd

CHECKINS = untappd.load_latest_checkins()

# todo :: much of this code is copied from category_by_date
# once I have nice smoother plots, ideally move functionality
# into a shared file

SHOW_IBUS = False

GROUP_TIMESPAN = datetime.timedelta(days=31 * 3)


def evaluate_over_time_periods(
    checkins: list[untappd.Checkin],
    map_func: Callable,
    combine_func: Callable,
    timespan: datetime.timedelta,
) -> tuple[list[datetime.datetime], list]:
    start_date = min(c.datetime for c in checkins)
    end_date = max(c.datetime for c in checkins) - timespan
    day_starts = [
        datetime.datetime.fromordinal(n)
        for n in range(start_date.toordinal(), end_date.toordinal())
    ]
    return (
        day_starts,
        [
            combine_func([
                result
                for ci in CHECKINS
                if (
                    start <= ci.datetime < start + timespan
                    and (result := map_func(ci)) is not None
                )
                # todo :: don't do this loop every time :P
            ])
            for start in day_starts
        ],
    )


day_starts, abvs = evaluate_over_time_periods(
    checkins=CHECKINS,
    map_func=lambda ci: ci.beer.abv,
    combine_func=statistics.mean,
    timespan=GROUP_TIMESPAN,
)

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
