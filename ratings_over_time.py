from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from untappd_shared import load_latest_json

print("loading")
data = load_latest_json()

# I think this is redundant but just to be sure
print("sorting")
data = sorted(
    data,
    key=lambda d: d["created_at"],
)

print("gathering data")
spread = 400
step = 20
frames = []
for start in range(0, len(data) - spread, step):
    checkins = data[start:start + spread]
    frames.append(
        {
            "ratings": Counter(
                round(float(ci["rating_score"] or "0") * 4)
                for ci in checkins
            ),
            "start_date": checkins[0]["created_at"].split()[0],  # date only
            "end_date": checkins[-1]["created_at"].split()[0],
        }
    )

print("setting up plots")
fig, ax = plt.subplots()
x_data = [i / 4 for i in range(1, 21)]
y_first_frame = [0 for i in range(1, 21)]
ln, = plt.plot(x_data, y_first_frame, 'ro')

def init():
    ax.set_xlim(0, 5)
    ax.set_ylim(0, spread / 2)
    return ln,

def update(frame):
    # todo :: include dates
    ax.set_title(frame["start_date"] + " ~ " + frame["end_date"])
    ln.set_data(
        x_data,
        [frame["ratings"].get(i) for i in range(1, 21)],
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
