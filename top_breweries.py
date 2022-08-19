"""
scores breweries based on their top beers and average beers,
depending on the average_score_weight :)
"""

from collections import defaultdict

import untappd

CHECKINS = untappd.load_latest_checkins()

brewery_checkins = defaultdict(list)

for c in CHECKINS:
    brewery_checkins[c.beer.brewery].append(c)


def score_checkin_list(checkins, dropoff_ratio=0.8, average_score_weight=0.5):
    """
    dropoff_ratio indicates how much to scale weighting for subsequent beers
    """
    # todo :: beer averaging should be library functionality
    arbb = untappd.average_rating_by_beer(checkins)
    top_ratings = sorted(
        arbb.items(),
        reverse=True,
        key = lambda t: t[1],
    )
    return (
        average_score_weight * sum(v for _, v in top_ratings) / len(top_ratings)
        + (1 - average_score_weight) * (1 - dropoff_ratio) * sum(  # geometric sum
            r * dropoff_ratio ** i
            for i, (_, r) in enumerate(top_ratings)
        ),
        top_ratings,
    )


scores_breweries = [
    (*score_checkin_list(checkins), brewery)
    for brewery, checkins in brewery_checkins.items()
]

scores_sorted = sorted(scores_breweries, key=lambda t: t[0], reverse=True)

SHOW_TOP_N = 5  # 0 for less detailed view :)

for i, (score, top_beers, brewery) in enumerate(scores_sorted[:20]):
    print(f"{i+1: <3} {score:.2f}  {brewery}")
    if SHOW_TOP_N > 0:
        print(f"{brewery}'s top {SHOW_TOP_N} beers:")
        for beer, rating in top_beers[:SHOW_TOP_N]:
            print(f"{rating:.2f} {beer}")
        print()
