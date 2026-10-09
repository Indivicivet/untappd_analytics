from collections import Counter
import datetime

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from matplotlib import patches

import untappd
import untappd_utils

MAX_GAP = datetime.timedelta(hours=8)
RATINGS = [x / 4 for x in range(1, 21)]


def _pca_ellipses(pairs, sigmas):
    coords = np.array([(i, j) for i in RATINGS for j in RATINGS])
    mean = np.average(coords, axis=0, weights=pairs.flat)
    cov = np.cov((coords - mean).T, aweights=pairs.flat)

    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    for sigma in sigmas:
        width_real, height_real = 2 * sigma * np.sqrt(eigenvals[order])
        angle = np.degrees(np.arctan2(eigenvecs[1, order[0]], eigenvecs[0, order[0]]))

        # convert center + widths into heatmap grid units
        step = RATINGS[1] - RATINGS[0]
        yield patches.Ellipse(
            xy=(
                (mean[0] - RATINGS[0]) / step,
                (mean[1] - RATINGS[0]) / step,
            ),
            width=width_real / step,
            height=height_real / step,
            angle=angle,
            edgecolor="white",
            facecolor="none",
            lw=1,
        )


@untappd_utils.show_or_save_to_out_file
def plot_consecutive_checkin_rating_correlation(checkins):
    freqs = Counter()
    for c0, c1 in zip(checkins, checkins[1:]):
        if (
            MAX_GAP is not None
            and c0.datetime is not None
            and c1.datetime is not None
            and not (datetime.timedelta(0) <= (c1.datetime - c0.datetime) <= MAX_GAP)
        ):
            continue
        freqs[(c0.rating, c1.rating)] += 1

    pairs = np.array([[freqs.get((i, j), 0) for j in RATINGS] for i in RATINGS])

    fig, ax = plt.subplots(figsize=(10, 8))
    seaborn.heatmap(
        pairs,
        xticklabels=RATINGS,
        yticklabels=RATINGS,
        annot=True,
        fmt="d",
        square=True,
        ax=ax,
    )
    ax.invert_yaxis()
    for ellipse in _pca_ellipses(pairs, [1, 2, 3]):
        ax.add_patch(ellipse)
    ax.axline((0, 0), (19, 19))

    plt.xlabel("first checkin")
    plt.ylabel("second checkin")


if __name__ == "__main__":
    CIS = untappd.load_latest_checkins()
    plot_consecutive_checkin_rating_correlation(CIS)
