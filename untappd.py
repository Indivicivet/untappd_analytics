import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

DEFAULT_DATA_SOURCE = Path("__file__").parent / "data_sources"


def load_latest_json(data_source=None):
    data_source = (
        DEFAULT_DATA_SOURCE
        if data_source is None
        else Path(data_source)
    )
    files = sorted(
        data_source.glob("*.json"),
        key=lambda x: x.stat().st_mtime, reverse=True,
    )
    if not files:
        raise Exception(f"Couldn't find any json files in {data_source}")
    data_unsorted = json.loads(files[0].read_text())
    # I think this is redundant but just to be sure
    # todo :: verify that :)
    return sorted(
        data_unsorted,
        key=lambda d: d["created_at"],
    )


CATEGORY_KEYWORDS = {
    "stout": ["stout"],
    "sour": ["sour", "lambic"],
    "ipa": ["ipa"],
    "lager": ["lager", "pilsner"],
    "other ale": ["ale"],
    "other": [],
}


def categorize(checkin_dict):
    type_str = checkin_dict["beer_type"].lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in type_str for keyword in keywords):
            return category
    return "other"


# todo :: slots?
@dataclass
class Beer:
    name: str
    brewery: str
    type: str = ""
    abv: str = ""  # todo type ?
    ibu: str = ""  # todo type ?
    url: str = ""


@dataclass
class Venue:
    name: str
    city: str
    state: str
    country: str
    # todo :: types for these :)
    lat: Optional[Any] = None
    long: Optional[Any] = None


@dataclass
class Checkin:
    beer: Beer
    comment: str
    url: str
    venue: Optional[Venue] = None

    @classmethod
    def from_dict(cls, d):
        maybe_venue = {
            "venue": Venue(
                name=d["venue_name"],
                city=d["venue_city"],
                state=d["venue_state"],
                country=d["venue_country"],
                lat=d["venue_lat"],
                long=d["venue_lng"],
            )
        } if d["venue_name"] is not None else {}
        return cls(
            beer=Beer(
                name=d["beer_name"],
                brewery=d["brewery_name"],
                type=d["beer_type"],
                abv=d["beer_abv"],
                ibu=d["beer_ibu"],
                url=d["beer_url"],
            ),
            comment=d["comment"],
            **maybe_venue,
            url=d["checkin_url"],
        )


def load_latest_checkins():
    # todo :: data source
    # todo :: csv option...? or warning about no csv :)
    data_dicts = load_latest_json()
    return [
        Checkin.from_dict(d)
        for d in data_dicts
    ]
