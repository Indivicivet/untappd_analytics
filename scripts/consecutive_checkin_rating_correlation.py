from collections import Counter

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from matplotlib import patches

import untappd

CIS = untappd.load_latest_checkins()

freqs = Counter()

for c0, c1 in zip(CIS, CIS[1:]):
    freqs[(c0.rating, c1.rating)] += 1

ratings = [x / 4 for x in range(1, 21)]
pairs = np.array([[freqs.get((i, j), 0) for j in ratings] for i in ratings])


def pca_ellipses(sigmas):
    coords = np.array([(i, j) for i in ratings for j in ratings])
    mean = np.average(coords, axis=0, weights=pairs.flat)
    cov = np.cov((coords - mean).T, aweights=pairs.flat)

    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    for sigma in sigmas:
        width_real, height_real = 2 * sigma * np.sqrt(eigenvals[order])
        angle = np.degrees(np.arctan2(eigenvecs[1, order[0]], eigenvecs[0, order[0]]))

        # convert center + widths into heatmap grid units
        step = ratings[1] - ratings[0]
        yield patches.Ellipse(
            xy=(
                (mean[0] - ratings[0]) / step,
                (mean[1] - ratings[0]) / step,
            ),
            width=width_real / step,
            height=height_real / step,
            angle=angle,
            edgecolor="white",
            facecolor="none",
            lw=1,
        )


# plot heatmap
fig, ax = plt.subplots()
seaborn.heatmap(
    pairs,
    xticklabels=ratings,
    yticklabels=ratings,
    annot=True,
    fmt="d",
    ax=ax,
)
ax.invert_yaxis()
for ellipse in pca_ellipses([1, 2, 3]):
    ax.add_patch(ellipse)

plt.xlabel("first checkin")
plt.ylabel("second checkin")
plt.show()
