import matplotlib.pyplot as plt
import seaborn

import untappd

CHECKINS = untappd.load_latest_checkins()

times = [checkin.datetime for checkin in CHECKINS]
times_non_taster = [
    checkin.datetime
    for checkin in CHECKINS
    if checkin.serving_type != "Taster"
]
times_unique = []
times_repeat = []
hit_beers = set()
for checkin in CHECKINS:
    if checkin.beer in hit_beers:
        times_repeat.append(checkin.datetime)
        continue
    times_unique.append(checkin.datetime)
    hit_beers.add(checkin.beer)

seaborn.set()

plt.figure(figsize=(12.8, 7.2))
plt.plot(
    times_non_taster,
    range(len(times_non_taster)),
    label="non-taster",
    linestyle="dashed"
)
plt.plot(times, range(len(times)), label="total checkins")
plt.plot(
    times_unique,
    range(len(times_unique)),
    label="unique",
    linestyle="dashed"
)
plt.plot(
    times_repeat,
    range(len(times_repeat)),
    label="repeat",
    linestyle="dashed",
)
plt.xlabel("date")
plt.ylabel("checkins")
plt.legend()
plt.show()
