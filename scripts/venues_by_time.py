import datetime
from collections import defaultdict, Counter

from matplotlib import pyplot as plt

import untappd
import untappd_utils

DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


@untappd_utils.show_or_save_to_out_file
def plot_venues_by_time(checkins):
    day_counts = defaultdict(lambda: Counter())

    for ci in checkins:
        day_counts[(ci.datetime - datetime.timedelta(hours=5)).strftime("%A")][
            ci.venue
        ] += 1

    rows = 2
    cols = 4
    _, axes = plt.subplots(rows, cols, figsize=(12.8, 7.2))
    for i, day in enumerate(DAYS):
        top_counts = day_counts[day].most_common(10)
        ax = axes[i // cols, i % cols]
        ax.pie(
            [c for _, c in top_counts],
            labels=[v if v is not None else "None" for v, _ in top_counts],
        )
        ax.set_title(day)


if __name__ == "__main__":
    CIS = untappd.load_latest_checkins()
    plot_venues_by_time(CIS)
