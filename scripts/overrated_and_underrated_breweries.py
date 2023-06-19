from collections import defaultdict, Counter
from pathlib import Path

import untappd

CHECKINS = untappd.load_latest_checkins()

brewery_checkins = defaultdict(list)

for c in CHECKINS:
    brewery_checkins[c.beer.brewery].append(c)

SHOW_N = 20
MIN_CHECKINS = 5

scores_personal_global = [
    (
        untappd.magic_rating(checkins)[0],
        global_rating,
        brewery,
    )
    for brewery, checkins in brewery_checkins.items()
    if (global_rating := untappd.magic_rating(checkins, use_global=True)[0]) != 0
        and len(checkins) > MIN_CHECKINS
]

print(f"using {MIN_CHECKINS=}")

scores_sorted = sorted(scores_personal_global, key=lambda t: t[0] - t[1])
print("most overrated breweries (in my opinion):")
for my_rating, global_rating, brewery in scores_sorted[:SHOW_N]:
    print(
        f"{brewery} || my rating: {my_rating:.2f}"
        f", global_rating: {global_rating:.2f}"
        f", delta = {global_rating - my_rating:.2f}"
    )

print()
print("most underrated breweries (in my opinion):")
for my_rating, global_rating, brewery in scores_sorted[-SHOW_N:][::-1]:
    print(
        f"{brewery} || my rating: {my_rating:.2f}"
        f", global_rating: {global_rating:.2f}"
        f", delta = {my_rating - global_rating:.2f}"
    )
