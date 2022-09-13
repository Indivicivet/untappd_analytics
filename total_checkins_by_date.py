import matplotlib.pyplot as plt

import untappd

CHECKINS = untappd.load_latest_checkins()

times = [checkin.datetime for checkin in CHECKINS]

plt.figure(figsize=(12.8, 7.2))
plt.plot(times, range(len(times)))
plt.show()
