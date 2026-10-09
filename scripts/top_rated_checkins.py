import matplotlib.pyplot as plt
import seaborn

import untappd
import untappd_utils

IGNORE_REPEAT_BEERS = True


@untappd_utils.show_or_save_to_out_file
def plot_top_rated_checkins(checkins):
    hit_beers = set()
    datetimes_by_rating = {
        4.5: [],
        4.25: [],
    }

    for ci in checkins:
        if ci.beer in hit_beers:
            continue
        if ci.rating in datetimes_by_rating:
            datetimes_by_rating[ci.rating].append(ci.datetime)
            # todo :: behaviour for 4.25->4.5 and 4.5->4.25?
            if IGNORE_REPEAT_BEERS:
                hit_beers.add(ci.beer)

    # todo :: consider merge with cumulative_checkins?
    seaborn.set()
    plt.figure(figsize=(12.8, 7.2))
    for rating, datetimes in datetimes_by_rating.items():
        plt.plot(datetimes, range(len(datetimes)), label=f"{rating} ratings")
    plt.legend()


if __name__ == "__main__":
    CHECKINS = untappd.load_latest_checkins()
    plot_top_rated_checkins(CHECKINS)
