from collections import defaultdict

import untappd

# todo :: consolidate this categorization logic
# into a utils file?
import rating_histogram_by_various

TOP_N = 5
CIS = untappd.load_latest_checkins()

festivals = defaultdict(list)
for ci in CIS:
    festivals[rating_histogram_by_various.festival_with_year(ci)].append(ci)

for festival, checkins in festivals.items():
    print(festival)
    print(f"Top {TOP_N}")
    # accept duplicate check-ins; probably rare at festivals
    for i, ci in enumerate(
        sorted(checkins, key=lambda c: c.rating, reverse=True)[:TOP_N]
    ):
        print(f"#{i + 1} ({ci.rating}): {ci.beer}")
    print()
