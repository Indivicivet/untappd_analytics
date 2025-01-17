"""
where did you go back for more?
"""

from collections import Counter, defaultdict

import untappd

CIS = untappd.load_latest_checkins()

places = defaultdict(Counter)
for ci in CIS:
    places[ci.venue][ci.beer] += 1

print(
    sorted(
        places.items(),
        key=lambda t: sum(t[1].values()),
        reverse=True,
    )[:20]
)
