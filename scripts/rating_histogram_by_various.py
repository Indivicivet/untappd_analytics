from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
import datetime
from typing import Optional, Any, Callable, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn

import untappd
import untappd_utils


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
    def top_n(cls, func, all_checkins, n=5, allow_other=False):
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


# todo :: move to untappd?
@untappd_utils.show_or_save_to_out_file
def show_histogram(
    data: Sequence[untappd.Checkin],
    # todo :: its really "sortable":
    func: Callable[[untappd.Checkin], Union[Any, list[Any], tuple[Any]]],
    normalize: bool = False,
    show_n_checkins: bool = True,
    title: Optional[str] = None,
):
    seaborn.set()
    category_data = defaultdict(lambda: defaultdict(int))
    for checkin in data:
        if (result := func(checkin)) is not None:
            rating = checkin.rating or 0
            if isinstance(result, (list, tuple)):
                # can't check iterable because strings are iterable
                for category in result:
                    category_data[category][rating] += 1
            else:
                category_data[result][rating] += 1

    x_data = [i / 4 for i in range(1, 21)]
    plt.figure(figsize=(12.8, 7.2))
    plt.gca().margins(0.01, 0.01)
    for label, counts in sorted(category_data.items()):
        scale_factor = 100 / sum(counts.values()) if normalize else 1
        y_data = [counts.get(x, 0) * scale_factor for x in x_data]
        total_checkins = sum(counts.values())
        average = sum(x * counts.get(x, 0) for x in x_data) / total_checkins
        plt.plot(
            *untappd_utils.smooth_ratings(x_data, y_data),
            label=(
                f"{label} ({total_checkins} checkins, {average:.3f} average)"
                if show_n_checkins
                else label
            )
        )
    plt.legend()
    plt.title(title)
    plt.xlabel("rating")
    plt.ylabel(
        "percentage of checkins\n(normalized per category)"
        if normalize
        else "number of checkins"
    )
    plt.gca().xaxis.set_minor_locator(mtick.MultipleLocator(0.25))
    plt.gca().grid(which="minor", color="w", linewidth=0.5)
    if normalize:
        plt.gca().yaxis.set_major_formatter(
            mtick.PercentFormatter(decimals=0),
        )


@untappd_utils.show_or_save_to_out_file
def show_violin(
    data: Sequence[untappd.Checkin],
    # todo :: its really "sortable":
    func: Callable[[untappd.Checkin], Union[Any, list[Any], tuple[Any]]],
    normalize: bool = False,
    show_n_checkins: bool = True,
    title: Optional[str] = None,
):
    seaborn.set()
    # similar code to show_histogram's accumulation, but here we actually
    # store the checkins, since this is what violinplot wants
    # (alternatively we could share code and convert format in one function...)
    category_data = defaultdict(list)
    for checkin in data:
        if (result := func(checkin)) is not None:
            rating = checkin.rating or 0
            if isinstance(result, (list, tuple)):
                # can't check iterable because strings are iterable
                for category in result:
                    category_data[category].append(rating)
            else:
                category_data[result].append(rating)

    plt.figure(figsize=(12.8, 7.2))
    plt.gca().margins(0.01, 0.01)
    plt.violinplot(
        list(category_data.values()),
        quantiles=[[0, 0.1, 0.5, 0.9, 1]] * len(category_data),
        vert=False,
        widths=0.75,
        bw_method=0.25 + 0.1,  # known discrete sampling, smoothed a bit extra
    )
    if plt.xlim()[0] < 2:
        plt.xlim(left=2)
    plt.title(title)
    plt.xlabel("rating")
    plt.gca().set_yticks(
        list(range(1, len(category_data) + 1)),
        labels=list(category_data),
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
    if (
        checkin.venue is not None
        and (
            "beer festival" in checkin.venue.name.lower()
            or "fest 20" in checkin.venue.name.lower()
        )
    ):
        return f"{checkin.venue.name} ({checkin.datetime.year})"
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


def save_various_plots(
    checkins,
    out_dir=None,
    violin=False,
):
    if out_dir is None:
        out_dir = Path(__file__).parent / "out"
    for tag, func in {
        "style_category": lambda checkin: checkin.beer.get_style_category(),
        "hour": lambda checkin: checkin.datetime.hour,
        "session_n": SessionTracker().session_n,
        "strength": strength_class,
        "date_seg_singapore": date_segment_sg,
        "date_seg_nederlands": date_segment_nl,
        "date_seg_3": date_segment_3,
        "date_seg_4_sos": date_segment_4,
        "festival": festival_with_year,
        "brewery_common": ByFuncSpecificValuesOnly.top_n(
            func=lambda ci: ci.beer.brewery.name,
            all_checkins=checkins,
            n=6,
        ),
        "brewery_top_magic": ByFuncSpecificValuesOnly.top_magic_n(
            func=lambda ci: ci.beer.brewery.name,
            all_checkins=checkins,
            n=6,
        ),
        "venue_common": ByFuncSpecificValuesOnly.top_n(
            func=lambda ci: ci.venue.name if ci.venue is not None else None,
            all_checkins=checkins,
        ),
        "venue_top_magic": ByFuncSpecificValuesOnly.top_magic_n(
            func=lambda ci: ci.venue.name if ci.venue is not None else None,
            all_checkins=checkins,
        ),
        "nth_time_having": BeerNthTimeTracker().beer_n,
        "taster_or_not": lambda checkin: checkin.serving_type == "Taster",
        "weak_strong_main_categories": weak_strong_main_categories,
        "checkin_comment_length": checkin_comment_length,
        "fruit": by_keyword([
            "cherry",
            "blueberry",
            "mango",
            # " apple",  # meh
            "pineapple",
            "banana",
            "passion",
            # "guava",
        ]),
        "hop": by_keyword([
            # "chinook",
            "mosaic",
            "simcoe",
            "cascade",
            "citra",
            "centennial",
            "motueka",
            "galaxy",  # beware: other beers may sneak in...
        ]),
    }.items():
        if violin:
            show_violin(
                checkins,
                func=func,
                title=f"ratings by {tag}",
                out_file=out_dir / f"violin_by_{tag}.png"
            )
        else:
            show_histogram(
                checkins,
                func=func,
                normalize=tag not in ["hour", "hop"],
                title=f"ratings by {tag}",
                out_file=out_dir / f"ratings_by_{tag}.png",
            )


if __name__ == "__main__":
    CHECKINS = untappd.load_latest_checkins()
    save_various_plots(
        CHECKINS,
        violin=True,
    )
    # show_histogram(CHECKINS, func=strength_class, normalize=True)
