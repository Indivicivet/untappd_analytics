import matplotlib.pyplot as plt
import seaborn as sns

import untappd
import untappd_utils


def get_adjusted_time(datetime_obj):
    # Map 7 AM to 0 to keep drinking sessions contiguous
    return (datetime_obj.hour - 7) % 24 + datetime_obj.minute / 60


def format_time(datetime_obj):
    return datetime_obj.strftime("%H:%M")


def get_quantiles(data, n_blocks):
    blocks = []
    total = len(data)
    for i in range(n_blocks):
        start = i * total // n_blocks
        end = (i + 1) * total // n_blocks
        blocks.append(data[start:end])
    return blocks


@untappd_utils.show_or_save_to_out_file
def plot_time_of_day_kdes(checkins, n_blocks=6):
    # Filter for check-ins with ABV
    checkins_with_abv = [c for c in checkins if c.beer.abv is not None]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.8, 14.4))
    sns.set_theme()

    # Subplot 1: ABV by Time-of-Day
    checkins_with_abv.sort(key=lambda c: get_adjusted_time(c.datetime))
    time_blocks = get_quantiles(checkins_with_abv, n_blocks)

    for block in time_blocks:
        abvs = [c.beer.abv for c in block]
        start_time = format_time(block[0].datetime)
        end_time = format_time(block[-1].datetime)
        label = f"{start_time} - {end_time} ({len(block)} checkins)"
        sns.kdeplot(abvs, label=label, fill=True, alpha=0.1, ax=ax1)

    ax1.set_title(
        f"ABV distribution by time of day (partitioned into {n_blocks} time quantiles)"
    )
    ax1.set_xlabel("ABV (%)")
    ax1.set_ylabel("Density")
    ax1.legend()

    # Subplot 2: Time-of-Day by ABV
    checkins_with_abv.sort(key=lambda c: c.beer.abv)
    abv_blocks = get_quantiles(checkins_with_abv, n_blocks)

    for block in abv_blocks:
        times = [get_adjusted_time(c.datetime) for c in block]
        start_abv = block[0].beer.abv
        end_abv = block[-1].beer.abv
        label = f"{start_abv:.1f}% - {end_abv:.1f}% ({len(block)} checkins)"
        sns.kdeplot(times, label=label, fill=True, alpha=0.1, ax=ax2)

    ax2.set_title(
        f"Time-of-day distribution by ABV (partitioned into {n_blocks} ABV quantiles)"
    )
    ax2.set_xlabel("Time of Day")
    ax2.set_ylabel("Density")

    # Format X-axis for Subplot 2 to show clock times
    ticks = [0, 6, 12, 18, 24]
    labels = ["07:00", "13:00", "19:00", "01:00", "07:00"]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels)
    ax2.set_xlim(0, 24)
    ax2.legend()

    plt.tight_layout()


if __name__ == "__main__":
    DATA = untappd.load_latest_checkins()
    plot_time_of_day_kdes(DATA, out_file="out/time_of_day_kdes.png")
