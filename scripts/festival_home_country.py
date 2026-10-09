import math
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

import untappd
import untappd_categorise
import untappd_utils

TOP_N = 5


@untappd_utils.show_or_save_to_out_file
def plot_festival_home_country(checkins):
    festivals = defaultdict(Counter)
    for ci in checkins:
        festivals[untappd_categorise.festival_with_year(ci)][
            ci.beer.brewery.country
        ] += 1

    cols = math.ceil(len(festivals) ** 0.5 * 1.3)  # heuristic based on aspect
    fig, axs = plt.subplots(
        math.ceil(len(festivals) / cols),
        cols,
        figsize=(12.8, 9),
    )
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, (festival_name, festival_data) in zip(axs.flat, festivals.items()):
        top_country, top_count = festival_data.most_common(1)[0]
        rest = festival_data.total() - top_count
        ax.set_title(festival_name, fontsize=10)
        ax.pie(
            [top_count, rest],
            labels=[
                f"{top_country} ({top_count / festival_data.total():.1%})",
                "other" if rest else "",
            ],
        )
        ax.set_visible(True)


if __name__ == "__main__":
    CIS = untappd.load_latest_checkins()
    plot_festival_home_country(CIS)
