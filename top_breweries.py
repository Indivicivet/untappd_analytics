"""
scores breweries based on their top beers :)
"""

from collections import defaultdict

import untappd

CHECKINS = untappd.load_latest_checkins()

brewery_checkins = defaultdict(list)

for c in CHECKINS:
    brewery_checkins[c.beer.brewery].append(c)


def score_checkin_list(checkins, dropoff_ratio=0.8):
    """
    dropoff_ratio indicates how much to scale weighting for subsequent beers
    """
    # todo :: beer averaging should be library functionality
    beer_ratings = defaultdict(list)
    for c in checkins:
        if c.rating is None:
            continue
        beer_ratings[c.beer].append(c.rating)
    ratings = [
        sum(rating_list) / len(rating_list)
        for rating_list in beer_ratings.values()
    ]
    return (1 - dropoff_ratio) * sum(  # geometric sum
        r * dropoff_ratio ** i
        for i, r in enumerate(sorted(ratings, reverse=True))
    )


scores_breweries = [
    (score_checkin_list(checkins), brewery)
    for brewery, checkins in brewery_checkins.items()
]

scores_sorted = sorted(scores_breweries, key=lambda t: t[0], reverse=True)

for i, (score, brewery) in enumerate(scores_sorted[:20]):
    print(f"{i+1: <3} {score:.2f}  {brewery}")
