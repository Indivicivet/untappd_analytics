from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn

import untappd
import untappd_utils

print("loading")
CHECKINS = untappd.load_latest_checkins()

print("gathering data")
CUMULATIVE = False
SPREAD = 400  # ignored for CUMULATIVE
STEP = 40
NUM_ONION_SKINS = 10


frames = []
slices = (
    [
        slice(0, pos)
        for pos in range(STEP, len(CHECKINS) - STEP, STEP)
    ]
    if CUMULATIVE else
    [
        slice(start, start + SPREAD)
        for start in range(0, len(CHECKINS) - SPREAD, STEP)
    ]
)
    
    
for i, ci_slice in enumerate(slices):
    checkins = CHECKINS[ci_slice]
    frames.append(
        {
            "num_ratings": len(checkins),
            "ratings": Counter(
                round(float(ci.rating or 0) * 4)
                for ci in checkins
            ),
            "start_date": checkins[0].datetime.date(),  # date only
            "end_date": checkins[-1].datetime.date(),
        }
    )


print("setting up plots")
seaborn.set()
fig, ax = plt.subplots(figsize=(12.8, 7.2))
x_data = [i / 4 for i in range(1, 21)]
y_first_frame = [0 for i in range(1, 21)]

data_queue = [
    (x_data, y_first_frame)
    for _ in range(NUM_ONION_SKINS + 1)
]
plot_lines = [
    plt.plot(*data, c=c0)[0]
    for data, c0 in zip(
        data_queue,
        [(v, v, v) for v in np.linspace(0.9, 0.5, len(data_queue)-1)]
        + [(0, 0, 0)],
    )
]
plt.ylabel("number of checkins")
plt.xlabel("rating")


def init():
    ax.set_xlim(0, 5)
    return plot_lines


def update(frame):
    ax.set_title(f'{frame["start_date"]} ~ {frame["end_date"]}')
    ax.set_ylim(0, frame["num_ratings"] * (0.3 if CUMULATIVE else 0.4))
    data_queue.pop(0)
    data_queue.append(
        untappd_utils.smooth_ratings(
            x_data,
            [frame["ratings"].get(i, 0) for i in range(1, 21)],
        ),
    )
    for line, data in zip(plot_lines, data_queue):
        line.set_data(*data)
    return plot_lines


print("animating")
ani = animation.FuncAnimation(
    fig,
    func=update,
    frames=frames,
    init_func=init,
    #blit=True,  # incompatible with ax.set_title()
    interval=100,
)
plt.show()
