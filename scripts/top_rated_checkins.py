import matplotlib.pyplot as plt
import seaborn

import untappd

CHECKINS = untappd.load_latest_checkins()

times45 = [
    ci.datetime
    for ci in CHECKINS
    if ci.rating == 4.5
]

times425 = [
    ci.datetime
    for ci in CHECKINS
    if ci.rating == 4.25
]

seaborn.set()
plt.figure(figsize=(12.8, 7.2))
plt.plot(times45, range(len(times45)), label="4.5 ratings")
plt.plot(times425, range(len(times425)), label="4.25 ratings")
plt.show()
