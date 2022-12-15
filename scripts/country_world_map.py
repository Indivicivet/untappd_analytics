from collections import defaultdict
from pathlib import Path

import numpy as np
import colour
import folium
import geopy.geocoders

import untappd
import filecached

CHECKINS = untappd.load_latest_checkins()
country_ratings = defaultdict(list)


# todo :: might want to use a slightly different set of
# magic_rating params for country-based?
MAGIC_RATING = True
rating_type_str = "magic" if MAGIC_RATING else "average"

for ci in CHECKINS:
    country_ratings[ci.beer.brewery.country].append(ci)

country_rating_and_count = sorted(
    (
        (
            (
                untappd.magic_rating(cis)[0]
                if MAGIC_RATING
                else sum(ci.rating for ci in cis) / len(cis)
            ),
            len(cis),
            country,
        )
        for country, cis in country_ratings.items()
    ),
    reverse=True,
)

world = folium.Map(tiles="cartodbpositron")
markers = folium.FeatureGroup().add_to(world)

geocoder = geopy.geocoders.Nominatim(user_agent="untappd_analytics")


def get_new_latlong(country):
    location = geocoder.geocode(country)
    return [location.latitude, location.longitude]


get_latlong = filecached.Function(
    func=get_new_latlong,
    file=Path("out/world_latlongs.json"),
)

for i, (rating, n_ratings, country) in enumerate(country_rating_and_count):
    country_rank_str = (
        f"{country} (rank {i+1})"
        f" - {rating_type_str} rating"
        f" {rating:.2f} over {n_ratings} ratings"
    )
    print(country_rank_str)
    rating_colour_range = (
        (2, 4.1)
        if MAGIC_RATING
        else (2.5, 4)
    )
    hue = np.interp(rating, rating_colour_range, (0, 0.3))
    folium.Circle(
        location=get_latlong(country),
        radius=20_000 * n_ratings ** 0.4,  # meters
        # stroke=False,
        color=colour.Color(hsl=(hue, 1, 0.3)).hex_l,
        fill=True,
        fill_color=colour.Color(hsl=(hue, 1, 0.5)).hex_l,
        tooltip=country_rank_str,
    ).add_to(markers)

world.render()
MAP_OUT_FILE = f"out/world_map_{rating_type_str}.html"
world.save(MAP_OUT_FILE)
print(f"saved map to {MAP_OUT_FILE}")
