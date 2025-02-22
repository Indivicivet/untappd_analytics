"""
where did you go back for more?
"""

from collections import Counter, defaultdict

import untappd

N_VENUES = 20

CIS = untappd.load_latest_checkins()

places = defaultdict(Counter)
for ci in CIS:
    places[ci.venue][ci.beer] += 1

for venue_i, (venue, beer_counts) in enumerate(sorted(
    places.items(),
    key=lambda t: sum(t[1].values()),
    reverse=True,
)[:N_VENUES]):
    print(f"#{venue_i + 1}", "No venue" if venue is None else venue.name)
    for beer_i, (beer, count) in enumerate(beer_counts.most_common(100)):
        if count <= 2:
            break
        print(f"#{beer_i + 1} ({count}): {beer}")
    print()
