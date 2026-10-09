# based on venues_by_time but plotting the opposite way around

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
COUNT_VISITS = True  # ignore dupes on the same day


@untappd_utils.show_or_save_to_out_file
def plot_venues_by_time_inverse(checkins):
    venue_counts = defaultdict(Counter)
    venue_total_counts = Counter()
    venue_counted_days = defaultdict(set)

    for ci in checkins:
        sensible_datetime = ci.datetime - datetime.timedelta(hours=5)
        if COUNT_VISITS and sensible_datetime.date() in venue_counted_days[ci.venue]:
            continue  # skip if dupe on same day
        venue_counts[ci.venue][sensible_datetime.strftime("%A")] += 1
        venue_total_counts[ci.venue] += 1
        venue_counted_days[ci.venue].add(sensible_datetime.date())

    rows = 2
    cols = 4
    _, axes = plt.subplots(rows, cols, figsize=(12.8, 7.2))
    for i, (venue, _) in enumerate(venue_total_counts.most_common(rows * cols)):
        count_dict = venue_counts[venue]
        ax = axes[i // cols, i % cols]
        ax.pie(
            [count_dict[day] for day in DAYS],
            labels=[f"{day} ({count_dict[day]})" for day in DAYS],
        )
        ax.set_title(venue)

    plt.suptitle(
        "Number of days visited" if COUNT_VISITS else "Total checkins on each day"
    )


if __name__ == "__main__":
    CIS = untappd.load_latest_checkins()
    plot_venues_by_time_inverse(CIS)
