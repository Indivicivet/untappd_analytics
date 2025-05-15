from collections import defaultdict, Counter

from matplotlib import pyplot as plt

import untappd

CIS = untappd.load_latest_checkins()

VENUE_TAGS = ["brewdog"]  # only include venues matching these


# venue -> datetime (period) -> first party and third party counts
venues = defaultdict(lambda: defaultdict(Counter))
for c in CIS:
    try:
        # reasonably safe to assume at most one ^^
        venue_tag = next(tag for tag in VENUE_TAGS if tag in str(c.venue).lower())
    except StopIteration:
        continue
    first_party = venue_tag in str(c.beer.brewery).lower()
    venues[c.venue][(c.datetime.year, c.datetime.month)][first_party] += 1
    venues[c.venue]["total"][first_party] += 1

for venue, data in venues.items():
    if data.pop("total").total() < 10:  # expunge from plot but check this
        continue
    datetime_periods = list(data.keys())
    r = range(len(datetime_periods))
    firsts_n = [cts[True] for cts in data.values()]
    plt.title(venue)
    plt.bar(r, firsts_n, label="First party")
    plt.bar(
        r,
        [cts[False] for cts in data.values()],
        bottom=firsts_n,
        label="Third party",
    )
    plt.xticks(r, datetime_periods, rotation=60, horizontalalignment="right")
    plt.legend()
    plt.show()
