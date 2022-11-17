import matplotlib.pyplot as plt
import seaborn

import untappd

CHECKINS = untappd.load_latest_checkins()

seaborn.set()

# todo :: would much rather <checkins in period> rather than cumulative
plt.figure(figsize=(12.8, 7.2))
for category in untappd.CATEGORY_KEYWORDS:
    results = [
        checkin.datetime
        for checkin in CHECKINS
        if checkin.beer.get_style_category() == category
    ]
    plt.plot(results, range(len(results)), label=category)
plt.xlabel("date")
plt.ylabel("checkins")
plt.legend()
plt.show()
