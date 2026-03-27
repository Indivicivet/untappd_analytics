import matplotlib.pyplot as plt
import seaborn as sns

import untappd
import untappd_utils


def get_adjusted_time(datetime_obj):
    # Map 5 AM to 0 to keep drinking sessions contiguous
    return (datetime_obj.hour - 5) % 24 + datetime_obj.minute / 60


def format_time(datetime_obj):
    return datetime_obj.strftime("%H:%M")


@untappd_utils.show_or_save_to_out_file
def plot_time_of_day_kdes(checkins, n_blocks=6):
    # Filter for check-ins with ABV
    checkins_with_abv = [c for c in checkins if c.beer.abv is not None]

    # Sort by adjusted time (5 AM start)
    checkins_with_abv.sort(key=lambda c: get_adjusted_time(c.datetime))

    # Split into n_blocks quantiles
    block_size = len(checkins_with_abv) // n_blocks
    blocks = [
        checkins_with_abv[i * block_size : (i + 1) * block_size]
        for i in range(n_blocks)
    ]

    plt.figure(figsize=(12.8, 7.2))
    sns.set_theme()

    for i, block in enumerate(blocks):
        abvs = [c.beer.abv for c in block]
        start_time = format_time(block[0].datetime)
        end_time = format_time(block[-1].datetime)
        label = f"{start_time} - {end_time} ({len(block)} checkins)"
        sns.kdeplot(abvs, label=label, fill=True, alpha=0.1)

    plt.title(
        f"ABV distribution by time of day (partitioned into {n_blocks} equal blocks)"
    )
    plt.xlabel("ABV (%)")
    plt.ylabel("Density")
    plt.legend()


if __name__ == "__main__":
    DATA = untappd.load_latest_checkins()
    plot_time_of_day_kdes(DATA, out_file="out/time_of_day_kdes.png")
