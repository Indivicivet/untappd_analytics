# todo :: can we do more interesting things with sentiment analysis? :)
# we probably can! but the basic thing this script does atm is subsumed by
# scatter_by_various.py

import matplotlib.pyplot as plt
from nltk.sentiment import vader

import untappd

CIS = untappd.load_latest_checkins()

analyzer = vader.SentimentIntensityAnalyzer()
for c in CIS:
    c._compound_score = analyzer.polarity_scores(c.comment)["compound"]


for c in sorted(CIS, key=lambda x: x._compound_score, reverse=True)[:25]:
    print(f"{c._compound_score} ({c.rating}) | {c.beer} | {c.comment}")

plt.figure(figsize=(12.8, 7.2))
plt.scatter(
    [c.rating for c in CIS],
    [c._compound_score for c in CIS],
    alpha=0.05,
    s=100,
)
plt.show()
