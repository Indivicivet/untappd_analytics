import matplotlib.pyplot as plt
import seaborn

import untappd

CHECKINS = untappd.load_latest_checkins()

times = [checkin.datetime for checkin in CHECKINS]

seaborn.set()

plt.figure(figsize=(12.8, 7.2))
plt.plot(times, range(len(times)))
plt.xlabel("date")
plt.ylabel("total checkins")
plt.show()
