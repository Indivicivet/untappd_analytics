"""
scores breweries based on their top beers and average beers,
depending on the average_score_weight :)
"""

from collections import defaultdict, Counter

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


scores_breweries = [
    (*untappd.magic_rating(checkins), brewery)
    for brewery, checkins in brewery_checkins.items()
]

scores_sorted = sorted(scores_breweries, key=lambda t: t[0], reverse=True)

SHOW_N_BREWERIES = 20
SHOW_TOP_N = 5  # 0 for less detailed view :)
SHOW_STYLES = True


all_beer_score, _ = untappd.magic_rating(CHECKINS)

for i, (score, top_beers, brewery) in enumerate(scores_sorted[:SHOW_N_BREWERIES]):
    print(f"#{i+1: <3} {brewery} (magic rating: {score:.2f})")
    if SHOW_STYLES:
        print(
            "Styles across all checkins:",
            ", ".join(
                f"{style.capitalize()} {ratio:.1%}"
                for style, ratio in get_style_ratios(brewery_checkins[brewery])
            )
        )
    if SHOW_TOP_N > 0:
        # print(f"{brewery}'s top {SHOW_TOP_N} beers:")
        for beer, rating in top_beers[:SHOW_TOP_N]:
            print(f"{rating:.2f}  {beer}")
        print()

print(f"(magic rating across all beers: {all_beer_score:.2f})")
