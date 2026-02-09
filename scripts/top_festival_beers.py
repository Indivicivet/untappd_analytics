from collections import defaultdict

import untappd

import untappd_categorise

TOP_N = 5
CIS = untappd.load_latest_checkins()

festivals = defaultdict(list)
for ci in CIS:
    festivals[untappd_categorise.festival_with_year(ci)].append(ci)

for festival, checkins in festivals.items():
    print(festival)
    print(f"Top {TOP_N}")
    # accept duplicate check-ins; probably rare at festivals
    for i, ci in enumerate(
        sorted(checkins, key=lambda c: c.rating, reverse=True)[:TOP_N]
    ):
        print(f"#{i + 1} ({ci.rating}): {ci.beer}")
    print()
