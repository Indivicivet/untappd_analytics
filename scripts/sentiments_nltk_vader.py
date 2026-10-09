# todo :: can we do more interesting things with sentiment analysis? :)
# we probably can! but the basic thing this script does atm is subsumed by
# scatter_by_various.py

import matplotlib.pyplot as plt
from nltk.sentiment import vader

import untappd
import untappd_utils


@untappd_utils.show_or_save_to_out_file
def plot_vader_sentiments(checkins):
    analyzer = vader.SentimentIntensityAnalyzer()
    for c in checkins:
        c._compound_score = analyzer.polarity_scores(c.comment)["compound"]

    for c in sorted(checkins, key=lambda x: x._compound_score, reverse=True)[:25]:
        print(f"{c._compound_score} ({c.rating}) | {c.beer} | {c.comment}")

    plt.figure(figsize=(12.8, 7.2))
    plt.scatter(
        [c.rating for c in checkins],
        [c._compound_score for c in checkins],
        alpha=0.05,
        s=100,
    )


if __name__ == "__main__":
    CIS = untappd.load_latest_checkins()
    plot_vader_sentiments(CIS)
