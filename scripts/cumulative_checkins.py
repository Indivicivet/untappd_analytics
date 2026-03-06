import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import seaborn
import numpy as np
from scipy.stats import gaussian_kde

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
def plot_extrapolated(use_times, color):
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
        color=color,
        linewidth=1,
        linestyle="dashed",
    )


def get_checkin_rate_curve(use_times, num_points=400):
    numeric_time_days = mpl_dates.date2num(use_times)
    numeric_time_for_plot = np.linspace(
        numeric_time_days[0],
        numeric_time_days[-1],
        num_points,
    )
    if len(use_times) < 2:
        return numeric_time_for_plot, np.zeros_like(numeric_time_for_plot)

    edge_window = min(len(use_times), 11)
    edge_spacing = np.diff(numeric_time_days[-edge_window:]).mean()
    edge_spacing = max(edge_spacing, 1e-3)
    edge_pad = 10
    padded_time_days = np.concatenate(
        [
            numeric_time_days,
            numeric_time_days[-1] + edge_spacing * np.arange(1, edge_pad + 1),
        ]
    )

    fitted_kde = gaussian_kde(padded_time_days)
    # gaussian_kde integrates to 1, so scale back to checkins/day.
    checkin_rate = fitted_kde(numeric_time_for_plot) * len(padded_time_days)

    edge_point_count = max(1, num_points // 20)
    checkin_rate[-edge_point_count:] = checkin_rate[-edge_point_count - 1]
    return numeric_time_for_plot, checkin_rate


def plot_rate_extrapolated(use_times, color):
    numeric_time_for_plot, checkin_rate = get_checkin_rate_curve(use_times)
    fit_window_start_time = use_times[-1] - FIT_MOST_RECENT
    fit_window_start_numeric = mpl_dates.date2num(fit_window_start_time)
    fit_window_mask = numeric_time_for_plot >= fit_window_start_numeric
    fitted_polynomial = np.poly1d(
        np.polyfit(
            numeric_time_for_plot[fit_window_mask],
            checkin_rate[fit_window_mask],
            deg=EXTRAPOLATE_DEGREE,
        )
    )
    extrapolated_numeric_time = np.linspace(
        fit_window_start_numeric,
        mpl_dates.date2num(use_times[-1] + EXTRAPOLATE_TIME),
        200,
    )
    extrapolated_rate = np.maximum(
        fitted_polynomial(extrapolated_numeric_time),
        0,
    )
    plt.plot(
        mpl_dates.num2date(extrapolated_numeric_time),
        extrapolated_rate,
        color=color,
        linewidth=1,
        linestyle="dashed",
    )


def plot_cumulative_checkins():
    seaborn.set()
    palette = seaborn.color_palette()
    plt.figure(figsize=(12.8, 7.2))
    plot_extrapolated(times_non_taster, color=palette[0])
    plt.plot(
        times_non_taster,
        range(len(times_non_taster)),
        label="non-taster",
        color=palette[0],
        alpha=0.7,
    )
    plot_extrapolated(times, color=palette[1])
    plt.plot(
        times,
        range(len(times)),
        label="total checkins",
        color=palette[1],
        alpha=0.7,
    )
    plot_extrapolated(times_unique, color=palette[2])
    plt.plot(
        times_unique,
        range(len(times_unique)),
        label="unique",
        color=palette[2],
        alpha=0.7,
    )
    plot_extrapolated(times_repeat, color=palette[3])
    plt.plot(
        times_repeat,
        range(len(times_repeat)),
        label="repeat",
        color=palette[3],
    )
    plt.xlabel("date")
    plt.ylabel("checkins")
    plt.legend()
    plt.show()


def plot_checkin_rate():
    seaborn.set()
    palette = seaborn.color_palette()
    plt.figure(figsize=(12.8, 7.2))

    plot_rate_extrapolated(times_non_taster, color=palette[0])
    numeric_time_for_plot, checkin_rate = get_checkin_rate_curve(times_non_taster)
    plt.plot(
        mpl_dates.num2date(numeric_time_for_plot),
        checkin_rate,
        label="non-taster",
        color=palette[0],
        alpha=0.7,
    )

    plot_rate_extrapolated(times, color=palette[1])
    numeric_time_for_plot, checkin_rate = get_checkin_rate_curve(times)
    plt.plot(
        mpl_dates.num2date(numeric_time_for_plot),
        checkin_rate,
        label="total checkins",
        color=palette[1],
        alpha=0.7,
    )

    plot_rate_extrapolated(times_unique, color=palette[2])
    numeric_time_for_plot, checkin_rate = get_checkin_rate_curve(times_unique)
    plt.plot(
        mpl_dates.num2date(numeric_time_for_plot),
        checkin_rate,
        label="unique",
        color=palette[2],
        alpha=0.7,
    )

    plot_rate_extrapolated(times_repeat, color=palette[3])
    numeric_time_for_plot, checkin_rate = get_checkin_rate_curve(times_repeat)
    plt.plot(
        mpl_dates.num2date(numeric_time_for_plot),
        checkin_rate,
        label="repeat",
        color=palette[3],
    )

    plt.xlabel("date")
    plt.ylabel("checkins / day")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plot_cumulative_checkins()
    plot_checkin_rate()
