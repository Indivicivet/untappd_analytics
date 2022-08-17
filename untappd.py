import csv
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Sequence

DEFAULT_DATA_SOURCE = Path("__file__").parent / "data_sources"


def load_latest_datafile(data_source=None):
    data_source = (
        DEFAULT_DATA_SOURCE
        if data_source is None
        else Path(data_source)
    )
    files = sorted(
        [*data_source.glob("*.json"), *data_source.glob("*.csv")],
        key=lambda x: x.stat().st_mtime, reverse=True,
    )
    if not files:
        raise Exception(f"Couldn't find any .json or .csv files in {data_source}")
    if files[0].suffix == ".json":
        data_unsorted = json.loads(files[0].read_text())
    elif files[0].suffix == ".csv":
        with open(files[0], encoding="utf-8") as f:
            csv_lines = csv.reader(f)
            headers = next(csv_lines)
            headers[0] = "beer_name"  # todo :: why does it seem corrupted...?
            data_unsorted = []
            for row in csv_lines:
                data_unsorted.append(dict(zip(headers, row)))
    else:
        raise Exception(f"somehow found unknown file extension for file {files[0]}")
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
            id=int(d["brewery_id"]) if d.get("brewery_id") else None,
        )

    def to_dict(self):
        return {
            "brewery_name": self.name,
            "brewery_url": self.url,
            "brewery_country": self.country,
            "brewery_city": self.city,
            "brewery_state": self.state,
            "brewery_id": str(self.id) if self.id is not None else None,
        }


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

    def __str__(self):
        return f"{self.name} | {self.brewery} | {self.type} ({self.abv:.1f}%)"

    @classmethod
    def from_checkin_dict(cls, d):
        try:
            ibu = int(d["beer_ibu"])
        except ValueError:
            ibu = float(d["beer_ibu"])
            warnings.warn(
                f"you have a beer with non-integer ibu {ibu}"
                ", that may break some assumptions..."
            )
        return cls(
            name=d["beer_name"],
            # todo :: cache breweries? (and beers, ofc)
            brewery=Brewery.from_checkin_dict(d),
            abv=float(d["beer_abv"]),
            id=int(d["bid"]),
            global_rating=d["global_rating_score"],
            global_weighted_rating=d["global_weighted_rating_score"],
            type=d["beer_type"],
            ibu=ibu,
            url=d["beer_url"],
        )

    def to_dict(self):
        return {
            "beer_name": self.name,
            **self.brewery.to_dict(),
            "beer_abv": str(self.abv),  # todo :: precision?
            "bid": str(self.id),
            "global_rating_score": self.global_rating,
            "global_weighted_rating_score": self.global_weighted_rating,
            "beer_type": self.type,
            "beer_ibu": str(self.ibu),
            "beer_url": str(self.url),
        }

    def get_style_category(self):
        type_str = self.type.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in type_str for keyword in keywords):
                return category
        return "other"


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

    def __str__(self):
        return f"{self.name} ({self.city})"

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

    def to_dict(self):
        return {
            "venue_name": self.name,
            "venue_city": self.city,
            "venue_state": self.state,
            "venue_country": self.country,
            "venue_lat": self.lat,
            "venue_lng": self.long,
        }


@dataclass
class Checkin:
    beer: Beer
    comment: str = ""
    url: Optional[str] = None
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

    def __hash__(self):
        # todo :: do we just want to always use one of these?
        if self.id is None:
            return hash(self.beer) + hash(self.comment) + hash(self.url)
        return self.id

    def _simple_str(self) -> str:
        return (
            f"Checkin of {self.beer.name} ({self.beer.brewery}), {self.beer.abv:.1f}%"
            + (f" | Rating: {self.rating} / 5" if self.rating else "")
            + (f" | At: {self.venue}" if self.venue else "")
            + (f' | Comment: "{self.comment}"' if self.comment else "")
        )

    @classmethod
    def from_dict(cls, d):
        maybe_venue = (
            {"venue": Venue.from_checkin_dict(d)}
            if d.get("venue_name") is not None
            else {}
        )
        return cls(
            beer=Beer.from_checkin_dict(d),
            comment=d["comment"],
            rating=float(d.get("rating_score") or 0) or None,  # todo...
            datetime=datetime.strptime(
                d["created_at"],
                "%Y-%m-%d %H:%M:%S"
            ) if d.get("created_at") else None,
            url=d["checkin_url"],
            flavour_profiles=[],  # todo :)
            # todo purchase_venue, serving_type
            id=int(d["checkin_id"]) if d.get("checkin_id") else None,
            photo_url=d["photo_url"],
            tagged_friends=d["tagged_friends"],
            total_toasts=int(d.get("total_toasts", 0) or 0),
            total_comments=int(d.get("total_comments", 0) or 0),
            **maybe_venue,
        )

    def to_dict(self):
        return {
            **self.beer.to_dict(),
            **(self.venue.to_dict() if self.venue is not None else {}),
            "comment": self.comment,
            "rating_score": str(self.rating) if self.rating is not None else None,
            "created_at": datetime.strftime(
                self.datetime,
                "%Y-%m-%d %H:%M:%S",
            ) if self.datetime is not None else "",
            "checkin_url": self.url,
            # todo :: something to do with flavour profiles :)
            "checkin_id": str(self.id) if self.id is not None else None,
            "photo_url": self.photo_url,
            "tagged_friends": self.tagged_friends,
            # todo :: so much duplication of this pattern, consider reshuffle
            "total_toasts": str(self.total_toasts) if self.total_toasts is not None else None,
            "total_comments": str(self.total_comments) if self.total_comments is not None else None,
        }


def load_latest_checkins():
    # todo :: data source
    # todo :: csv option...? or warning about no csv :)
    data_dicts = load_latest_datafile()
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
