import matplotlib.pyplot as plt

import untappd

CIS = untappd.load_latest_checkins()

plt.figure(figsize=(12.8, 7.2))
plt.scatter(
    [len(ci.comment) for ci in CIS],
    [ci.rating for ci in CIS],
    alpha=0.05,
    s=100,
)
plt.xlabel("comment length")
plt.ylabel("rating")
plt.show()
