from collections import defaultdict, Counter
from pathlib import Path

import untappd

CHECKINS = untappd.load_latest_checkins()

brewery_checkins = defaultdict(list)

for c in CHECKINS:
    brewery_checkins[c.beer.brewery].append(c)


def get_style_ratios(checkins):
    style_counts = Counter(c.beer.get_style_category() for c in checkins)
    total = style_counts.total()
    return [
        (style, val / total)
        for style, val in sorted(style_counts.items(), key=lambda t: t[1], reverse=True)
    ]


scores_personal_global = [
    (
        my_rating := untappd.magic_rating(checkins)[0],
        global_rating := untappd.magic_rating(checkins, use_global=True)[0],
        brewery,
    )
    for brewery, checkins in brewery_checkins.items()
]

SHOW_N = 10

scores_sorted = sorted(scores_personal_global, key=lambda t: t[0] - t[1])
print("most overrated breweries (in my opinion):")
for my_rating, global_rating, brewery in scores_sorted[:SHOW_N]:
    print(
        f"{brewery} || my rating: {my_rating}, global_rating: {global_rating}"
        f", delta = {global_rating - my_rating}"
    )

print()
print("most underrated breweries (in my opinion):")
for my_rating, global_rating, brewery in scores_sorted[-SHOW_N:][::-1]:
    print(
        f"{brewery} || my rating: {my_rating}, global_rating: {global_rating}"
        f", delta = {my_rating - global_rating}"
    )
