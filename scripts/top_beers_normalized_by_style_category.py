"""
initially a copy-paste of top_checkins_normalized_by_time_window.py
"""
# todo :: consolidate with top_checkins_normalized_by_time_window.py
# (although need to think about that being checkins, and this being beers...)
# todo :: also (optionally) mod out by ABV

import statistics
from collections import defaultdict

import untappd
import untappd_utils

CHECKINS = untappd.load_latest_checkins()
SKIP_OTHER = True


all_mean, all_std = untappd_utils.mean_and_std(CHECKINS)

# calculate statistics to normalize against
# note: compared to normalized by time, here we go by beer (not by checkin)
# making it different just for fun :) (and here I think it's slightly better)
ratings_by_beer = untappd.average_rating_by_beer(CHECKINS)
ratings_by_style_category = defaultdict(list)
for beer, rating in ratings_by_beer.items():
    ratings_by_style_category[beer.get_style_category()].append(rating)

stats_by_cats = {
    cat: (statistics.mean(ratings), statistics.stdev(ratings))
    for cat, ratings in ratings_by_style_category.items()
}


for beer, rating in ratings_by_beer.items():
    cat = beer.get_style_category()
    if SKIP_OTHER and "other" in cat:
        beer._normalized_rating = -1
        continue
    style_mean, style_std = stats_by_cats[cat]
    beer._normalized_rating = (
        all_mean + all_std * (rating - style_mean) / style_std
    )

print(f"beers normalized by style category")
print()
beers_sorted = sorted(
    ratings_by_beer,
    key=lambda beer: beer._normalized_rating,
    reverse=True,
)
for i, beer in enumerate(beers_sorted[:20]):
    rating = ratings_by_beer[beer]
    print(
        f"#{i + 1}, rating {rating:.3f} normalized {beer._normalized_rating:.3f}"
        f" with style category {beer.get_style_category()}"
    )
    print(beer)
    print()
