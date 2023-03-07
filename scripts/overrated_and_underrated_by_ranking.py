"""
initially copy-paste from overrated_and_underrated.py
because I want to display slightly different things for the ranking case
(and also the code just generally looks fairly different...)
"""

import untappd

SHOW_N = 20

CHECKINS = untappd.load_latest_checkins()
BEER_RATINGS = {
    beer: rating
    for beer, rating in untappd.average_rating_by_beer(CHECKINS).items()
    if beer.global_rating > 0
}


def get_rankings(beer, func):
    rankings = {}
    # note: this doesn't deal well with ties :(
    for i, beer in enumerate(sorted(beer, key=func, reverse=True)):
        rankings[beer] = i + 1
    return rankings


GLOBAL_RANKINGS = get_rankings(
    BEER_RATINGS.keys(),
    func=lambda beer: beer.global_rating,
)
PERSONAL_RANKINGS = get_rankings(
    BEER_RATINGS.keys(),
    func=lambda beer: BEER_RATINGS[beer],
)

sorted_beers = sorted(
    BEER_RATINGS.keys(),
    key=lambda beer: GLOBAL_RANKINGS[beer] - PERSONAL_RANKINGS[beer],
)

print(f"total beer count (hence lowest ranking): #{len(BEER_RATINGS)}")
print()

print("beers I rank much higher than average")
for beer in reversed(sorted_beers[-SHOW_N:]):
    print(
        f"{beer}"
        f" (ranking #{PERSONAL_RANKINGS[beer]}"
        f" vs global ranking #{GLOBAL_RANKINGS[beer]}"
        f" with ratings {BEER_RATINGS[beer]} vs {beer.global_rating})"
    )

print()
print("beers I rank much lower than average")
for beer in sorted_beers[:SHOW_N]:
    print(
        f"{beer}"
        f" (ranking #{PERSONAL_RANKINGS[beer]}"
        f" vs global ranking #{GLOBAL_RANKINGS[beer]}"
        f" with ratings {BEER_RATINGS[beer]} vs {beer.global_rating})"
    )
