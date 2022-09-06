import datetime

import matplotlib.pyplot as plt
import numpy as np

import untappd


CHECKINS = untappd.load_latest_checkins()

START_TIME = datetime.datetime(2022, 9, 3, 12, 0, 0)
END_TIME = datetime.datetime(2022, 9, 5, 23, 59, 59)

relevant_checkins = [c for c in CHECKINS if START_TIME < c.datetime < END_TIME]


TIME_STEPS = 1000
DETOX_PER_HOUR = 1.0
DETOX_PER_SECOND = DETOX_PER_HOUR / 3600
ASSUMED_BEER_VOLUME = 0.15  # * % gives volume!

time_offsets = np.linspace(0, (END_TIME - START_TIME).seconds, TIME_STEPS)
intox = 0
intoxes = [intox]

for t1, t2 in zip(time_offsets, time_offsets[1:]):
    # could make it O(n) assuming sorting but... shrug
    for ci in relevant_checkins:
        if t1 < (ci.datetime - START_TIME).seconds < t2:
            # todo :: tasters
            intox += ci.beer.abv * ASSUMED_BEER_VOLUME
    if intox > 0:
        intox -= DETOX_PER_SECOND * (t2 - t1)
    intoxes.append(intox)


plt.xlabel("seconds elapsed (todo: better!)")
plt.ylabel("units in body")
plt.plot(time_offsets, intoxes)
plt.show()
