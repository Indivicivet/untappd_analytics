from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
import datetime
from typing import Optional, Any, Callable, Sequence

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


# todo :: move to untappd?
def show_histogram(
    data: Sequence[untappd.Checkin],
    func: Callable[[untappd.Checkin], Any],  # todo :: its really "sortable"
    normalize: bool = False,
    # when `show_n_checkins` is not specified, show iff `normalize`
    show_n_checkins: Optional[bool] = None,
    title: Optional[str] = None,
    out_file: Optional[Path] = None,
):
    if show_n_checkins is None:
        show_n_checkins = normalize

    seaborn.set()
    category_data = defaultdict(lambda: defaultdict(int))
    for checkin in data:
        if (category := func(checkin)) is not None:
            category_data[category][checkin.rating or 0] += 1

    x_data = [i / 4 for i in range(1, 21)]
    plt.figure(figsize=(12.8, 7.2))
    for label, counts in sorted(category_data.items()):
        scale_factor = 100 / sum(counts.values()) if normalize else 1
        y_data = [counts.get(x, 0) * scale_factor for x in x_data]
        plt.plot(
            *untappd_utils.smooth_ratings(x_data, y_data),
            label=(
                f"{label} ({sum(counts.values())} checkins)"
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
    if out_file is None:
        plt.show()
    else:
        out_file = Path(out_file)
        out_file.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_file)


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


def by_brewery_popular_only(checkin):
    name = checkin.beer.brewery.name
    if any(x in name.lower() for x in [
        "amundsen",
        "vault city",
        "vocation",
        "siren",
        "wild beer",
        "nerdbrewing",
        # "moersleutel",
        # "brew york",
        # "northern monk",
        # "brouwerij kees",
    ]):
        return name
    return "other"


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


def save_various_plots(checkins, out_dir=None):
    if out_dir is None:
        out_dir = Path(__file__).parent / "out"
    for tag, func in {
        "style_category": lambda checkin: checkin.beer.get_style_category(),
        "hour": lambda checkin: checkin.datetime.hour,
        "session_n": SessionTracker().session_n,
        "strength": strength_class,
        "singapore": date_segment_sg,
        "nederlands": date_segment_nl,
        "brewery": by_brewery_popular_only,
        "nth_time_having": BeerNthTimeTracker().beer_n,
        "taster_or_not": lambda checkin: checkin.serving_type == "Taster",
        "weak_strong_main_categories": weak_strong_main_categories,
    }.items():
        out_file = out_dir / f"ratings_by_{tag}.png"
        show_histogram(
            checkins,
            func=func,
            normalize=tag not in ["hour"],
            title=f"ratings by {tag}",
            out_file=out_file,
        )
        print(f"saved ratings by {tag} to {out_file}")


CHECKINS = untappd.load_latest_checkins()
save_various_plots(CHECKINS)
# show_histogram(CHECKINS, func=strength_class, normalize=True)
