import csv
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Sequence, Union, Tuple

DEFAULT_DATA_SOURCE = Path(__file__).resolve().parent.parent / "data_sources"


def load_latest_datafile(
    data_source: Optional[Union[Path, str]] = None,
    prefer_non_sample_data: bool = True,
):
    data_source = (
        DEFAULT_DATA_SOURCE
        if data_source is None
        else Path(data_source)
    )
    files = sorted(
        [*data_source.glob("*.json"), *data_source.glob("*.csv")],
        key=lambda x: x.stat().st_mtime, reverse=True,
    )
    if prefer_non_sample_data:
        non_sample_files = [file for file in files if "sample" not in file.name]
        if any(non_sample_files):
            files = non_sample_files
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
    "wheat": ["wheat"],
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
    def from_checkin_dict(cls, d: dict):
        return cls(
            name=d["brewery_name"],
            url=d["brewery_url"],
            country=d["brewery_country"],
            city=d["brewery_city"],
            state=d["brewery_state"],
            id=int(d["brewery_id"]) if d.get("brewery_id") else None,
        )

    def to_dict(self) -> dict:
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
    abv: float  # todo :: would be nice if optional
    id: Optional[int] = None
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
    def from_checkin_dict(cls, d: dict):
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
            id=int(d["bid"]) if d.get("bid") else None,
            global_rating=d["global_rating_score"],
            global_weighted_rating=d["global_weighted_rating_score"],
            type=d["beer_type"],
            ibu=ibu,
            url=d["beer_url"],
        )

    def to_dict(self) -> dict:
        return {
            "beer_name": self.name,
            **self.brewery.to_dict(),
            "beer_abv": str(self.abv),  # todo :: precision?
            "bid": str(self.id) if self.id is not None else None,
            "global_rating_score": self.global_rating,
            "global_weighted_rating_score": self.global_weighted_rating,
            "beer_type": self.type,
            "beer_ibu": str(self.ibu),
            "beer_url": str(self.url),
        }

    def get_style_category(self) -> str:
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
        if self.city is None:
            return hash(f"VENUE {self.name} VENUE")
        return hash(self.name + self.city)

    def __str__(self):
        return f"{self.name} ({self.city})"

    @classmethod
    def from_checkin_dict(cls, d: dict):
        return cls(
            name=d["venue_name"],
            city=d["venue_city"],
            state=d["venue_state"],
            country=d["venue_country"],
            lat=d["venue_lat"],
            long=d["venue_lng"],
        )

    def to_dict(self) -> dict:
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
    # todo :: should actually make a decision about if these things are optional
    # or not -- we probably prefer it if we can assume they are present,
    # and they're present in the untappd data download
    datetime: Optional[datetime] = None
    venue: Optional[Venue] = None
    flavour_profiles: list[str] = field(default_factory=list)
    purchase_venue: Optional[str] = None  # todo :: type? Venue?
    serving_type: Optional[str] = None  # todo :: type? (enum for servings)
    id: Optional[int] = None
    photo_url: Optional[str] = None
    tagged_friends: Optional[str] = None  # todo :: type?
    total_toasts: int = 0
    total_comments: int = 0

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
    def from_dict(cls, d: dict):
        maybe_venue = (
            {"venue": Venue.from_checkin_dict(d)}
            if d.get("venue_name") is not None
            else {}
        )
        return cls(
            beer=Beer.from_checkin_dict(d),
            comment=d["comment"],
            rating=(
                float(d["rating_score"])
                if d["rating_score"]
                else None
            ),
            datetime=datetime.strptime(
                d["created_at"],
                "%Y-%m-%d %H:%M:%S"
            ) if d.get("created_at") else None,
            url=d["checkin_url"],
            flavour_profiles=[],  # todo :)
            purchase_venue=d["purchase_venue"] or None,  # todo :: make actual Venue?
            serving_type=d["serving_type"] or None,
            id=int(d["checkin_id"]) if d.get("checkin_id") else None,
            photo_url=d["photo_url"],
            tagged_friends=d["tagged_friends"],
            total_toasts=int(d.get("total_toasts", 0) or 0),
            total_comments=int(d.get("total_comments", 0) or 0),
            **maybe_venue,
        )

    def to_dict(self) -> dict:
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
            "purchase_venue": self.purchase_venue,
            "serving_type": self.serving_type,
            # todo :: something to do with flavour profiles :)
            "checkin_id": str(self.id) if self.id is not None else None,
            "photo_url": self.photo_url,
            "tagged_friends": self.tagged_friends,
            # todo :: so much duplication of this pattern, consider reshuffle
            "total_toasts": str(self.total_toasts) if self.total_toasts is not None else None,
            "total_comments": str(self.total_comments) if self.total_comments is not None else None,
        }


def load_latest_checkins(
    ignore_unrated: bool = True,
    ignore_tasters: bool = False,
) -> list[Checkin]:
    # todo :: data source
    # todo :: csv option...? or warning about no csv :)
    data_dicts = load_latest_datafile()
    return [
        Checkin.from_dict(d)
        for d in data_dicts
        if (
            not (ignore_unrated and d["rating_score"] == "")
            and not (ignore_tasters and d["serving_type"] == "Taster")
        )
    ]


def average_rating_by_beer(
    checkins: Sequence[Checkin],
    use_global: bool = False,
) -> dict[Beer, float]:
    """
    takes a sequence of checkins,
    returns a map from Beer to average checkin rating
    """
    beer_ratings = defaultdict(list)
    for c in checkins:
        if c.rating is None:
            continue
        rating = c.beer.global_rating if use_global else c.rating
        if rating != 0:
            beer_ratings[c.beer].append(rating)
    return {
        beer: sum(rating_list) / len(rating_list)
        for beer, rating_list in beer_ratings.items()
        if rating_list
    }


def magic_rating(
    checkins: Sequence[Checkin],
    dropoff_ratio: float = 0.8,
    average_score_weight: float = 0.5,
    neutral_rating: float = 3.25,
    use_global: bool = False,
) -> Tuple[float, list[Beer]]:
    """
    dropoff_ratio indicates how much to scale weighting for subsequent beers

    using either dropoff_ratio=1 or average_score_weight=1 should give
    a pure average rating
    todo :: unit test for this!

    `neutral_rating` indicates the rating at which additional beers give
    zero added score, and beers below this reduce the score.
    neutral_rating = 0 will give zero penalization for adding beers,
    regardless of rating.

    returns (magic score, list of beers by average rating)
    """
    top_ratings = sorted(
        average_rating_by_beer(checkins, use_global=use_global).items(),
        reverse=True,
        key=lambda t: t[1],
    )
    if not top_ratings:
        return 0, []
    return (
        # magic score
        average_score_weight * sum(v for _, v in top_ratings) / len(top_ratings)
        + (1 - average_score_weight) * (
            neutral_rating
            + (1 - dropoff_ratio) * sum(  # geometric sum
                (r - neutral_rating) * dropoff_ratio ** i
                for i, (_, r) in enumerate(top_ratings)
            )
        ),
        # list of beers by top rating
        top_ratings,
    )
