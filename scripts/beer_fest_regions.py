from collections import Counter

import untappd

FEST_TAG = "beer celebration"

ALL_CIS = untappd.load_latest_checkins()
FEST_CIS = [
    ci
    for ci in ALL_CIS
    if FEST_TAG in ((ci.venue and ci.venue.name) or "").lower()
]

country_freq = Counter(
    ci.beer.brewery.country
    for ci in FEST_CIS
)
print(country_freq)
