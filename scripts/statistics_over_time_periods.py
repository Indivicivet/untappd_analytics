import datetime
import statistics
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import seaborn

import untappd

CHECKINS = untappd.load_latest_checkins()


# todo :: much of this code is copied from category_by_date
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


def mean_plus_minus_std(values) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    return mean - std, mean, mean + std


def plot_statistics_over_time_periods(
    checkins: list[untappd.Checkin],
    map_func: Callable,
    y_label: Optional[str] = None,
    combine_func: Callable = mean_plus_minus_std,
    value_labels: Optional[list[str]] = None,
    out_file: Optional[Path] = None,
):
    if value_labels is None and combine_func == mean_plus_minus_std:
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
    plt.xlabel("start date")
    if y_label is not None:
        plt.ylabel(y_label)
    if out_file is None:
        plt.show()
    else:
        out_file = Path(out_file)
        out_file.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_file)


if __name__ == "__main__":
    plot_statistics_over_time_periods(
        checkins=CHECKINS,
        map_func=lambda ci: ci.beer.abv,
        y_label="abv",
        out_file=Path(__file__).parent / "out" / "statistics_over_time_abv.png",
    )
