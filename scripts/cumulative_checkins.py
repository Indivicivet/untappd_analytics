import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import seaborn
import numpy as np

import untappd

FIT_MOST_RECENT = datetime.timedelta(days=50)
EXTRAPOLATE_TIME = datetime.timedelta(days=90)
EXTRAPOLATE_DEGREE = 1

CHECKINS = untappd.load_latest_checkins()

times = [checkin.datetime for checkin in CHECKINS]
times_non_taster = [
    checkin.datetime
    for checkin in CHECKINS
    if checkin.serving_type != "Taster"
]
times_unique = []
times_repeat = []
hit_beers = set()
for checkin in CHECKINS:
    if checkin.beer in hit_beers:
        times_repeat.append(checkin.datetime)
        continue
    times_unique.append(checkin.datetime)
    hit_beers.add(checkin.beer)


# extrapolation
def plot_extrapolated(use_times, label_tag):
    numeric_time_days = mpl_dates.date2num(use_times)
    cumulative_counts = np.arange(len(use_times))
    fit_window_start_time = use_times[-1] - FIT_MOST_RECENT
    fit_window_mask = np.array(use_times) >= fit_window_start_time
    fitted_polynomial = np.poly1d(
        np.polyfit(
            numeric_time_days[fit_window_mask],
            cumulative_counts[fit_window_mask],
            deg=EXTRAPOLATE_DEGREE,
        )
    )
    numeric_time_for_plot = np.linspace(
        mpl_dates.date2num(fit_window_start_time),
        mpl_dates.date2num(use_times[-1] + EXTRAPOLATE_TIME),
        200,
    )
    plt.plot(
        mpl_dates.num2date(numeric_time_for_plot),
        fitted_polynomial(numeric_time_for_plot),
        label=f"{label_tag} (deg {EXTRAPOLATE_DEGREE})",
        linewidth=2,
    )


seaborn.set()
plt.figure(figsize=(12.8, 7.2))
plt.plot(
    times_non_taster,
    range(len(times_non_taster)),
    label="non-taster",
    linestyle="dashed",
)
plot_extrapolated(times_non_taster, label_tag="non-taster")
plt.plot(times, range(len(times)), label="total checkins")
plot_extrapolated(times, label_tag="total")
plt.plot(
    times_unique,
    range(len(times_unique)),
    label="unique",
    linestyle="dashed",
)
plot_extrapolated(times_unique, label_tag="unique")
plt.plot(
    times_repeat,
    range(len(times_repeat)),
    label="repeat",
    linestyle="dashed",
)
plt.xlabel("date")
plt.ylabel("checkins")
plt.legend()
plt.show()
