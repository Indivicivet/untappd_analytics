import datetime
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import seaborn

import untappd
import untappd_utils

CHECKINS = untappd.load_latest_checkins()


# todo :: much of this code is copied from style_frequency_over_time_periods
# once I have nice smoother plots, ideally move functionality
# into a shared file
def evaluate_over_time_periods(
    checkins: list[untappd.Checkin],
    map_func: Callable,
    combine_func: Callable,
    timespan: datetime.timedelta = datetime.timedelta(days=31 * 3),
) -> tuple[list[datetime.datetime], list]:
    start_date = min(c.datetime for c in checkins)
    end_date = max(c.datetime for c in checkins) - timespan
    day_starts = [
        datetime.datetime.fromordinal(n)
        for n in range(start_date.toordinal(), end_date.toordinal())
    ]
    return (
        day_starts,
        [
            combine_func([
                result
                for ci in CHECKINS
                if (
                    start <= ci.datetime < start + timespan
                    and (result := map_func(ci)) is not None
                )
                # todo :: don't do this loop every time :P
            ])
            for start in day_starts
        ],
    )


def percentiles(
    values,
    take_ratios=(0.1, 0.25, 0.5, 0.75, 0.9),
):
    sorted_values = sorted(values)
    return tuple(
        sorted_values[int(ratio * len(values))]
        for ratio in take_ratios
    )


# todo :: definitely very much speed optimisation possible ^^
@untappd_utils.show_or_save_to_out_file
def plot_statistics_over_time_periods(
    checkins: list[untappd.Checkin],
    map_func: Callable,
    y_label: Optional[str] = None,
    combine_func: Callable = untappd_utils.mean_plus_minus_std,
    value_labels: Optional[list[str]] = None,
    out_file: Optional[Path] = None,
):
    if value_labels is None and combine_func == untappd_utils.mean_plus_minus_std:
        value_labels = ["mean minus 1 std", "mean", "mean plus 1 std"]
    day_starts, various_stats = evaluate_over_time_periods(
        checkins=checkins,
        map_func=map_func,
        combine_func=combine_func,
    )
    seaborn.set()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(
        day_starts,
        various_stats,
        label=(
            value_labels
            if value_labels is not None
            else [None for _ in various_stats[0]]
        ),
    )
    plt.legend()
    plt.title(f"statistics of {y_label} over time periods")
    plt.xlabel("start date")
    if y_label is not None:
        plt.ylabel(y_label)


if __name__ == "__main__":
    for tag, func in {
        "abv": lambda ci: ci.beer.abv,
        "ibu": lambda ci: ci.beer.ibu if ci.beer.ibu > 0 else None,
        "rating": lambda ci: ci.rating,
    }.items():
        plot_statistics_over_time_periods(
            checkins=CHECKINS,
            map_func=func,
            y_label=tag,
            # combine_func=percentiles,
            # value_labels=["10%", "25%", "median", "75%", "90%"],
            out_file=Path(__file__).parent / "out" / f"statistics_over_time_{tag}.png",
        )
