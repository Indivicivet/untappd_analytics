import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Sequence

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


@dataclass
class Brewery:
    name: str
    url: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    id: Optional[int] = None

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    @classmethod
    def from_checkin_dict(cls, d):
        return cls(
            name=d["brewery_name"],
            url=d["brewery_url"],
            country=d["brewery_country"],
            city=d["brewery_city"],
            state=d["brewery_state"],
            id=int(d["brewery_id"]),
        )


# todo :: slots?
@dataclass
class Beer:
    name: str
    brewery: Brewery
    abv: float
    id: int
    global_rating: float = 0.0
    global_weighted_rating: float = 0.0
    type: str = ""
    ibu: Optional[float] = None
    url: str = ""

    def __hash__(self):
        return hash(self.name + self.brewery.name)

    @classmethod
    def from_checkin_dict(cls, d):
        return cls(
            name=d["beer_name"],
            # todo :: cache breweries? (and beers, ofc)
            brewery=Brewery.from_checkin_dict(d),
            abv=float(d["beer_abv"]),
            id=int(d["bid"]),
            global_rating=float(d["global_rating_score"]),
            global_weighted_rating=float(d["global_weighted_rating_score"]),
            type=d["beer_type"],
            ibu=float(d["beer_ibu"]),
            url=d["beer_url"],
        )


@dataclass
class Venue:
    name: str
    city: str
    state: str
    country: str
    # todo :: types for these :)
    lat: Optional[Any] = None
    long: Optional[Any] = None

    def __hash__(self):
        return hash(self.name + self.city)

    @classmethod
    def from_checkin_dict(cls, d):
        return cls(
            name=d["venue_name"],
            city=d["venue_city"],
            state=d["venue_state"],
            country=d["venue_country"],
            lat=d["venue_lat"],
            long=d["venue_lng"],
        )


@dataclass
class Checkin:
    beer: Beer
    comment: str
    url: str
    rating: Optional[float] = None
    datetime: Optional[datetime] = None
    venue: Optional[Venue] = None
    flavour_profiles: list[str] = field(default_factory=list)
    purchase_venue: Optional[str] = None
    id: Optional[int] = None
    photo_url: Optional[str] = None
    tagged_friends: Optional[str] = None  # todo :: type?
    total_toasts: Optional[int] = None
    total_comments: Optional[int] = None

    @classmethod
    def from_dict(cls, d):
        maybe_venue = (
            {"venue": Venue.from_checkin_dict(d)}
            if d["venue_name"] is not None
            else {}
        )
        return cls(
            beer=Beer.from_checkin_dict(d),
            comment=d["comment"],
            rating=float(d["rating_score"] or 0) or None,  # todo...
            datetime=datetime.strptime(
                d["created_at"],
                "%Y-%m-%d %H:%M:%S"
            ),
            url=d["checkin_url"],
            flavour_profiles=[],  # todo :)
            # todo purchase_venue, serving_type
            id=int(d["checkin_id"]),
            photo_url=d["photo_url"],
            tagged_friends=d["tagged_friends"],
            total_toasts=int(d["total_toasts"]),
            total_comments=int(d["total_comments"]),
            **maybe_venue,
        )


def load_latest_checkins():
    # todo :: data source
    # todo :: csv option...? or warning about no csv :)
    data_dicts = load_latest_json()
    return [
        Checkin.from_dict(d)
        for d in data_dicts
    ]


def average_rating_by_beer(checkins: Sequence[Checkin]) -> dict[Beer, float]:
    """
    takes a sequence of checkins,
    returns a map from Beer to average checkin rating
    """
    beer_ratings = defaultdict(list)
    for c in checkins:
        if c.rating is None:
            continue
        beer_ratings[c.beer].append(c.rating)
    return {
        beer: sum(rating_list) / len(rating_list)
        for beer, rating_list in beer_ratings.items()
    }
