import untappd

SHOW_N = 20

CHECKINS = untappd.load_latest_checkins()
BEER_RATINGS = untappd.average_rating_by_beer(CHECKINS)


sorted_beers = sorted(
    filter(
        lambda t: t[0].global_rating > 0,
        BEER_RATINGS.items(),
    ),
    key=lambda t: t[1] - t[0].global_rating,
)

print("beers I like much more than average")
for beer, beer_rating in reversed(sorted_beers[-SHOW_N:]):
    print(f"{beer} ({beer_rating} vs global rating {beer.global_rating})")

print()
print("beers I like much less than average")
for beer, beer_rating in sorted_beers[:SHOW_N]:
    print(f"{beer} ({beer_rating} vs global rating {beer.global_rating})")
