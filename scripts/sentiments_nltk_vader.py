import matplotlib.pyplot as plt
from nltk.sentiment import vader

import untappd

CIS = untappd.load_latest_checkins()

analyzer = vader.SentimentIntensityAnalyzer()

plt.figure(figsize=(12.8, 7.2))
plt.scatter(
    [c.rating for c in CIS],
    [analyzer.polarity_scores(c.comment)["compound"] for c in CIS],
    alpha=0.05,
    s=100,
)
plt.show()
