from collections import defaultdict
from pathlib import Path

import numpy as np
import colour
import folium
import geopy.geocoders

import untappd
import filecached

CHECKINS = untappd.load_latest_checkins()
place_ratings = defaultdict(list)


# todo :: might want to use a slightly different set of
# magic_rating params for country-based?
USE_VENUE = True  # instead of country
MAGIC_RATING = True and (not USE_VENUE)  # forbid being silly :)))
rating_type_str = "magic" if MAGIC_RATING else "average"

for ci in CHECKINS:
    place_ratings[
        ci.venue
        if USE_VENUE
        else ci.beer.brewery.country
    ].append(ci)

place_rating_and_count = sorted(
    (
        (
            (
                untappd.magic_rating(cis)[0]
                if MAGIC_RATING
                else sum(ci.rating for ci in cis) / len(cis)
            ),
            len(cis),
            place,
        )
        for place, cis in place_ratings.items()
        if place is not None
    ),
    key=lambda t: t[:2],
    reverse=True,
)

world = folium.Map(tiles="cartodbpositron")
markers = folium.FeatureGroup().add_to(world)

geocoder = geopy.geocoders.Nominatim(user_agent="untappd_analytics")


def get_new_latlong(country):
    location = geocoder.geocode(country)
    return [location.latitude, location.longitude]


get_latlong = (
    (lambda venue: [venue.lat, venue.long])
    if USE_VENUE
    else filecached.Function(
        func=get_new_latlong,
        file=Path("out/world_latlongs.json"),
    )
)

for i, (rating, n_ratings, place) in enumerate(place_rating_and_count):
    place_rank_str = (
        f"{place} (rank {i + 1})"
        f" - {rating_type_str} rating"
        f" {rating:.2f} over {n_ratings} ratings"
    )
    print(place_rank_str)
    rating_colour_range = (
        (2, 4.1)
        if MAGIC_RATING
        else (2.5, 3.75)
        if USE_VENUE
        else (2.5, 4)
    )
    hue = np.interp(rating, rating_colour_range, (0, 0.3))
    folium.Circle(
        location=get_latlong(place),
        radius=(
            20 * min(n_ratings, 20) ** 0.4  # * 4
            # ^make this 4x bigger if you haven't had a lot of beers somewhere!
            if USE_VENUE
            else 20_000 * n_ratings ** 0.4
        ),  # meters
        # stroke=False,
        color=colour.Color(hsl=(hue, 1, 0.3)).hex_l,
        fill=True,
        fill_color=colour.Color(hsl=(hue, 1, 0.5)).hex_l,
        tooltip=place_rank_str,
    ).add_to(markers)

world.render()
place_tag = "venues" if USE_VENUE else "countries"
MAP_OUT_FILE = (
        Path(__file__).parent / "out"
        / f"world_map_{place_tag}_{rating_type_str}.html"
)
world.save(MAP_OUT_FILE)
print(f"saved map to {MAP_OUT_FILE}")


try:
    from selenium import webdriver
except NameError:
    print("not launching since selenium not installed")
else:
    driver = webdriver.Chrome()
    driver.get("file://" + str(MAP_OUT_FILE.resolve()))
    input("press enter/exit to close browser...")
