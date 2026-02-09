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
import untappd_categorise


CHECKIN_VALUES = tuple(i / 4 for i in range(1, 21))


def _checkins_extra_str(rating_counts: Union[dict, list]) -> str:
    if not isinstance(rating_counts, dict):
        rating_counts = Counter(rating_counts)
    total_checkins = sum(rating_counts.values())
    average = sum(x * rating_counts.get(x, 0) for x in CHECKIN_VALUES) / total_checkins
    return f"({total_checkins} checkins, {average:.3f} average)"


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

    plt.figure(figsize=(12.8, 7.2))
    plt.gca().margins(0.01, 0.01)
    for label, counts in sorted(category_data.items()):
        scale_factor = 100 / sum(counts.values()) if normalize else 1
        y_data = [counts.get(x, 0) * scale_factor for x in CHECKIN_VALUES]
        plt.plot(
            *untappd_utils.smooth_ratings(CHECKIN_VALUES, y_data),
            label=(
                f"{label} {_checkins_extra_str(counts)}"
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
        labels=(
            [
                f"{label}\n{_checkins_extra_str(counts)}"
                for label, counts in category_data.items()
            ]
            if show_n_checkins
            else list(category_data)
        ),
    )




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
        "session_n": untappd_categorise.SessionTracker().session_n,
        "strength": untappd_categorise.strength_class,
        "date_seg_singapore": untappd_categorise.date_segment_sg,
        "date_seg_nederlands": untappd_categorise.date_segment_nl,
        "date_seg_3": untappd_categorise.date_segment_3,
        "date_seg_4_sos": untappd_categorise.date_segment_4,
        "date_seg_5": untappd_categorise.date_segment_5,
        "festival": untappd_categorise.festival_with_year,
        "brewery_common": untappd_categorise.ByFuncSpecificValuesOnly.most_common_n(
            func=lambda ci: ci.beer.brewery.name,
            all_checkins=checkins,
            n=6,
        ),
        "brewery_top_magic": untappd_categorise.ByFuncSpecificValuesOnly.top_magic_n(
            func=lambda ci: ci.beer.brewery.name,
            all_checkins=checkins,
            n=6,
        ),
        "venue_common": untappd_categorise.ByFuncSpecificValuesOnly.most_common_n(
            func=lambda ci: ci.venue.name if ci.venue is not None else None,
            all_checkins=checkins,
        ),
        "venue_top_magic": untappd_categorise.ByFuncSpecificValuesOnly.top_magic_n(
            func=lambda ci: ci.venue.name if ci.venue is not None else None,
            all_checkins=checkins,
        ),
        "nth_time_having": untappd_categorise.BeerNthTimeTracker().beer_n,
        "taster_or_not": lambda checkin: checkin.serving_type == "Taster",
        "weak_strong_main_categories": untappd_categorise.weak_strong_main_categories,
        "checkin_comment_length": untappd_categorise.checkin_comment_length,
        "fruit": untappd_categorise.by_keyword([
            "cherry",
            "blueberry",
            "mango",
            # " apple",  # meh
            "pineapple",
            "banana",
            "passion",
            # "guava",
        ]),
        "hop": untappd_categorise.by_keyword([
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
    for violin in [True, False]:
        save_various_plots(
            CHECKINS,
            violin=violin,
        )
        # show_histogram(CHECKINS, func=strength_class, normalize=True)
