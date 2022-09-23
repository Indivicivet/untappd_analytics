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

for ci in CHECKINS:
    country_ratings[ci.beer.brewery.country].append(ci.rating)

country_avg_and_count = sorted(
    (
        (sum(ratings) / len(ratings), len(ratings), country)
        for country, ratings in country_ratings.items()
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

for i, (average_rating, n_ratings, country) in enumerate(country_avg_and_count):
    country_rank_str = (
        f"{country} (rank {i+1})"
        f" - average {average_rating:.2f} over {n_ratings} ratings"
    )
    print(country_rank_str)
    red_rating = 2.5
    green_rating = 4.25
    hue = np.interp(average_rating, (red_rating, green_rating), (0, 1 / 3))
    folium.Circle(
        location=get_latlong(country),
        radius=30_000 * n_ratings ** 0.3,  # meters
        # stroke=False,
        color=colour.Color(hsl=(hue, 1, 0.3)).hex_l,
        fill=True,
        fill_color=colour.Color(hsl=(hue, 1, 0.5)).hex_l,
        tooltip=country_rank_str,
    ).add_to(markers)

world.render()
MAP_OUT_FILE = "out/world_map.html"
world.save(MAP_OUT_FILE)
print(f"saved map to {MAP_OUT_FILE}")
