from collections import Counter, defaultdict

from matplotlib import pyplot as plt

import untappd

FEST_TAG = "beer celebration"

ALL_CIS = untappd.load_latest_checkins()
FEST_CIS = [
    ci
    for ci in ALL_CIS
    if FEST_TAG in ((ci.venue and ci.venue.name) or "").lower()
]

ratings_by_country = defaultdict(list)
for ci in FEST_CIS:
    ratings_by_country[ci.beer.brewery.country].append(ci.rating)
country_freq = Counter(
    ci.beer.brewery.country
    for ci in FEST_CIS
)
print(country_freq.most_common())  # todo :: pie

country_order = [c for c, _ in country_freq.most_common()][::-1]  # top to bottom

fig, (ax_pie, ax_violin) = plt.subplots(1, 2, figsize=(12, 7.2))

ax_pie.pie(
    [country_freq[c] for c in country_order],
    labels=country_order,
    startangle=90,
)

# taken from rating_histogram_by_various.show_violin()
ax_violin.violinplot(
    list([ratings_by_country[country] for country in country_order]),
    quantiles=[[0, 0.1, 0.5, 0.9, 1]] * len(country_order),
    vert=False,
    widths=0.75,
    bw_method=0.25 + 0.1,  # known discrete sampling, smoothed a bit extra
)
ax_violin.set_yticks(
    list(range(1, len(country_order) + 1)),
    labels=list(country_order),
)
plt.show()
