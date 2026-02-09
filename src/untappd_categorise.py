from collections import Counter, defaultdict
from dataclasses import dataclass
import datetime
from typing import Optional, Any, Callable, Sequence, Union
import untappd


class SessionTracker:
    def __init__(self, cap=5, max_gap=datetime.timedelta(hours=2)):
        self.session = [untappd.Checkin(beer=None, datetime=datetime.datetime.min)]
        self.cap = cap
        self.max_gap = max_gap

    def session_n(self, checkin):
        if checkin.datetime > self.session[-1].datetime + self.max_gap:
            self.session = []
        self.session.append(checkin)
        n = len(self.session)
        if n >= self.cap:
            return f"{self.cap}+"
        return str(n)


class BeerNthTimeTracker:
    def __init__(self, max_n=4, separate_lager=False):
        self.beer_checkins = Counter()
        self.max_n = max_n
        self.separate_lager = separate_lager

    def beer_n(self, checkin):
        if self.separate_lager and checkin.beer.get_style_category() == "lager":
            return "lager"
        self.beer_checkins[checkin.beer] += 1
        n = self.beer_checkins[checkin.beer]
        if n < self.max_n:
            return str(n)
        return f"{self.max_n}+"


@dataclass
class ByFuncSpecificValuesOnly:
    func: Callable[[untappd.Checkin], str]
    included_values: Sequence[str]
    allow_other: bool = False

    def __call__(self, checkin):
        f_result = self.func(checkin)  # category, etc
        if f_result in self.included_values:
            return f_result
        if self.allow_other:
            return "other"
        return None

    @classmethod
    def most_common_n(cls, func, all_checkins, n=5, allow_other=False):
        return cls(
            func=func,
            included_values=[
                category
                for category, _ in Counter(
                    func(checkin)
                    for checkin in all_checkins
                    if func(checkin) is not None
                ).most_common(n)
            ],
            allow_other=allow_other,
        )

    @classmethod
    def top_magic_n(
        cls, func, all_checkins, n=5, allow_other=False, **magic_kwargs
    ):
        by_func_value = defaultdict(list)
        for checkin in all_checkins:
            by_func_value[func(checkin)].append(checkin)
        magics = [
            (untappd.magic_rating(checkins, **magic_kwargs)[0], func_val)
            for func_val, checkins in by_func_value.items()
        ]
        return cls(
            func=func,
            included_values=[
                func_val
                for _, func_val in sorted(magics, reverse=True, key=lambda t: t[0])
                if func_val is not None
            ][:n],
            allow_other=allow_other,
        )


def strength_class(checkin):
    if checkin.beer.abv < 5:
        return "5%"
    if checkin.beer.abv < 7.5:
        return "< 7.5%"
    if checkin.beer.abv < 10:
        return "< 10%"
    return "10%+"


def date_segment_sg(checkin):
    if checkin.datetime < datetime.datetime(2022, 6, 14, 20):
        return "Pre-Singapore"
    if checkin.datetime < datetime.datetime(2022, 7, 7):
        return "Singapore"
    return "Post-Singapore"


def date_segment_nl(checkin):
    if (
        datetime.datetime(2022, 9, 3)
        <= checkin.datetime
        <= datetime.datetime(2022, 9, 5)
    ):
        return "Netherlands"
    return "Otherwise"


def date_segment_3(checkin):
    if checkin.datetime < datetime.datetime(2023, 3, 25):
        return "Before"
    if checkin.datetime < datetime.datetime(2023, 5, 1, 16):
        return "During"
    return "After"


def date_segment_4(checkin):
    if checkin.datetime < datetime.datetime(2023, 11, 11):
        return None
    if checkin.datetime < datetime.datetime(2023, 11, 17):
        return "S"
    if checkin.datetime < datetime.datetime(2023, 11, 21, 4):
        return "O"
    if checkin.datetime < datetime.datetime(2023, 11, 21, 16):
        return "S"
    return None


def date_segment_5(checkin):
    if (
        datetime.datetime(2024, 9, 24)
        <= checkin.datetime
        <= datetime.datetime(2024, 10, 17)
    ):
        return "During"
    return "Other"


# todo :: there are better (non-lineplot) visualizations for this
def weak_strong_main_categories(checkin, threshold=7):
    style = checkin.beer.get_style_category()
    if style in ["stout", "sour", "ipa"]:
        strength = (
            f"{threshold}%+"
            if checkin.beer.abv >= threshold
            else f"<{threshold}%"
        )
        return f"{style}, {strength}"
    return None


def festival_with_year(checkin, include_non_festival=True):
    if checkin.venue is not None:
        if (
            " fest" in checkin.venue.name.lower()
            and "festival market" not in checkin.venue.name.lower()
        ) or "beer celebration" in checkin.venue.name.lower():
            return f"{checkin.venue.name}\n({checkin.datetime.year})"
        # hardcoded known beer festivals:
        if "Shiinoki Cul" in checkin.venue.name:
            return f"Kanazawa Craft Beer Festival\n({checkin.datetime.year})"
        if "Osaka Castle Park" in checkin.venue.name:
            return f"Craft Beer Holiday\n({checkin.datetime.year})"
    if include_non_festival:
        return "Non-festival"
    return None


def by_keyword(keywords):
    return lambda ci: [
        kw
        for kw in keywords
        if kw.lower() in str(ci).lower()
    ]


def checkin_comment_length(
    checkin,
    char_thresholds=(20, 50, 90),
):
    if len(checkin.comment) == 0:
        return "0"
    for prev, thresh in zip((0, ) + char_thresholds, char_thresholds):
        if len(checkin.comment) <= thresh:
            return f"{prev+1} ~ {thresh}"
    return f"{max(char_thresholds) + 1}+"
