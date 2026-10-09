import datetime
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import seaborn

import untappd
import untappd_utils

GROUP_TIMESPAN = datetime.timedelta(days=31 * 4)

PERCENTAGES = False
SKIP_OTHER = True


@untappd_utils.show_or_save_to_out_file
def plot_style_frequency_over_time_periods(checkins):
    start_date = min(c.datetime for c in checkins)
    end_date = max(c.datetime for c in checkins) - GROUP_TIMESPAN
    day_starts = [
        datetime.datetime.fromordinal(n)
        for n in range(start_date.toordinal(), end_date.toordinal())
    ]

    category_counts = []
    for start in day_starts:
        counts_this_time = Counter()
        for ci in checkins:
            if start <= ci.datetime < start + GROUP_TIMESPAN:
                counts_this_time[ci.beer.get_style_category()] += 1
                counts_this_time["total"] += 1
        category_counts.append(counts_this_time)

    seaborn.set()

    plt.figure(figsize=(12.8, 7.2))
    for category in untappd.CATEGORY_KEYWORDS:
        if SKIP_OTHER and "other" in category:
            continue
        plt.plot(
            day_starts,
            [
                counts[category] * (100 / counts["total"] if PERCENTAGES else 1)
                for counts in category_counts
            ],
            label=category,
        )
    plt.title(f"checkins by style ({GROUP_TIMESPAN.days} days following date)")
    plt.xlabel("date")
    plt.ylabel("checkin %" if PERCENTAGES else "checkins")
    plt.legend()


if __name__ == "__main__":
    CHECKINS = untappd.load_latest_checkins()
    plot_style_frequency_over_time_periods(CHECKINS)
