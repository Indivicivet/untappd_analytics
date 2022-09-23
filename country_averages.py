from collections import defaultdict

import colour
import folium
import geopy.geocoders

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

world = folium.Map(tiles="cartodbpositron")
markers = folium.FeatureGroup().add_to(world)
# todo :: do we need to cache these reqs so don't get banned? probably.
geolocator = geopy.geocoders.Nominatim(user_agent="untappd_analytics")

for i, (average_rating, n_ratings, country) in enumerate(country_avg_and_count):
    print(f"{i+1}: {country} - average {average_rating:.2f} over {n_ratings} ratings")
    loc = geolocator.geocode(country)
    red_rating = 2.5
    green_rating = 4.25
    hue = 0.333 * min(max((average_rating - red_rating) / (green_rating - red_rating), 0), 1)
    folium.Circle(
        location=(loc.latitude, loc.longitude),
        radius=30_000 * n_ratings ** 0.3,  # meters
        # stroke=False,
        color=colour.Color(hsl=(hue, 1, 0.3)).hex_l,
        fill=True,
        fill_color=colour.Color(hsl=(hue, 1, 0.5)).hex_l,
    ).add_to(markers)

world.render()
world.save("out/world.html")