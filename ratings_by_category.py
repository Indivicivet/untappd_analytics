from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from untappd_shared import load_latest_json, categorize

data = load_latest_json()

category_data = defaultdict(lambda: [0] * 20)
for checkin in data:
    rating_n = round(float(checkin["rating_score"] or "0") * 4)
    category_data[categorize(checkin)][rating_n - 1] += 1

x_data = [i / 4 for i in range(1, 21)]
for label, y_data in category_data.items():
    plt.plot(x_data, y_data, label=label)
plt.legend()
plt.show()
