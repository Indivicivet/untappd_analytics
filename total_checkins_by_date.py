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

seaborn.set()

plt.figure(figsize=(12.8, 7.2))
plt.plot(
    times_non_taster,
    range(len(times_non_taster)),
    label="non-taster",
    linestyle="dashed"
)
plt.plot(times, range(len(times)), label="total checkins")
plt.xlabel("date")
plt.ylabel("checkins")
plt.legend()
plt.show()
