from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import untappd
import untappd_utils

print("loading")
CHECKINS = untappd.load_latest_checkins()

CUMULATIVE = False

print("gathering data")
SPREAD = 400  # ignored for CUMULATIVE
STEP = 20
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
fig, ax = plt.subplots(figsize=(12.8, 7.2))
x_data = [i / 4 for i in range(1, 21)]
y_first_frame = [0 for i in range(1, 21)]
ln, = plt.plot(x_data, y_first_frame)
plt.ylabel("number of checkins")
plt.xlabel("rating")

def init():
    ax.set_xlim(0, 5)
    return ln,

def update(frame):
    ax.set_title(f'{frame["start_date"]} ~ {frame["end_date"]}')
    ax.set_ylim(0, frame["num_ratings"] * (0.3 if CUMULATIVE else 0.4))
    ln.set_data(
        untappd_utils.smooth_ratings(
            x_data,
            [frame["ratings"].get(i, 0) for i in range(1, 21)],
        ),
    )
    return ln,

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
