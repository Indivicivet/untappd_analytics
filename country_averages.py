from collections import defaultdict

import untappd

CHECKINS = untappd.load_latest_checkins()
country_ratings = defaultdict(list)

for ci in CHECKINS:
    country_ratings[ci.beer.brewery.country].append(ci.rating)

country_avg_and_count = sorted(
    (
        (sum(ratings) / len(ratings), len(ratings), country)
        for country, ratings in country_ratings.items()
    ),
    reverse=True,
)

for i, (average_rating, n_ratings, country) in enumerate(country_avg_and_count):
    print(f"{i+1}: {country} - average {average_rating:.2f} over {n_ratings} ratings")
