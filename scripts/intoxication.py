import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn

import untappd


CHECKINS = untappd.load_latest_checkins()

START_TIME = datetime.datetime(2022, 9, 3, 16, 0, 0)
END_TIME = datetime.datetime(2022, 9, 6, 6, 0, 0)

relevant_checkins = [c for c in CHECKINS if START_TIME < c.datetime < END_TIME]


TIME_STEPS = 1000
DETOX_PER_HOUR = 1.0
DETOX_PER_SECOND = DETOX_PER_HOUR / 3600
ASSUMED_BEER_VOLUME = 0.15  # * % gives volume!
TASTER_VOLUME = 0.05
PURCHASED_BEER_VOLUME = 0.33  # bottles

SHOW_RATINGS = False

time_offsets = np.linspace(0, (END_TIME - START_TIME).total_seconds(), TIME_STEPS)
total_consumption = 0
intox = 0
intoxes = [intox]

for t1, t2 in zip(time_offsets, time_offsets[1:]):
    # could make it O(n) assuming sorting but... shrug
    for ci in relevant_checkins:
        # todo :: could be more robust at only hitting a beer once
        if t1 <= (ci.datetime - START_TIME).total_seconds() < t2:
            print(f"{ci.datetime}, {ci.beer}, {ci.rating}")
            consumed_units = ci.beer.abv * (
                TASTER_VOLUME
                if ci.serving_type and ci.serving_type.lower() == "taster"  # todo :: attr?
                else PURCHASED_BEER_VOLUME
                if ci.purchase_venue
                else ASSUMED_BEER_VOLUME
            )
            total_consumption += consumed_units
            intox += consumed_units
    if intox > 0:
        intox -= DETOX_PER_SECOND * (t2 - t1)
        intox = max(intox, 0)
    intoxes.append(intox)


if not SHOW_RATINGS:
    seaborn.set()  # todo :: default seaborn.set() makes 2 axes a bit gross
fig, axes_intox = plt.subplots(figsize=(12.8, 7.2))
axes_intox.plot(time_offsets / 3600, intoxes, label="intoxication")
plt.title(
    f"total consumption = {total_consumption:.2f} units"
    f", over {(END_TIME - START_TIME) / datetime.timedelta(hours=1):.2f} hours"
)
plt.xlabel("hours elapsed (todo: better!)")
axes_intox.set_ylabel("units in body")
if SHOW_RATINGS:
    # todo :: add option of intox vs rating scatter
    axes_ratings = axes_intox.twinx()
    axes_ratings.plot(
        [(ci.datetime - START_TIME).total_seconds() / 3600 for ci in relevant_checkins],
        [ci.rating for ci in relevant_checkins],
        "ro",
        label="checkin rating",
    )
    axes_ratings.set_ylabel("rating")
    fig.legend()  # no point with just the line graph
plt.show()
