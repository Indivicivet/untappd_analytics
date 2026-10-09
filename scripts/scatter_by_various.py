from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import seaborn

try:
    from nltk.sentiment import vader
except ImportError:
    vader = None

import untappd
import untappd_utils


seaborn.set()


@untappd_utils.show_or_save_to_out_file
def scatter_things(
    dict_of_things,
    key_x,
    key_y,
    alpha=0.05,
    s=100,
):
    plt.figure(figsize=(12.8, 7.2))
    plt.scatter(
        dict_of_things[key_x],
        dict_of_things[key_y],
        alpha=alpha,
        s=s,
    )
    plt.xlabel(key_x)
    plt.ylabel(key_y)


def save_various_scatters(checkins, out_dir=None):
    if out_dir is None:
        out_dir = Path(__file__).parent / "out"

    dict_of_things = {
        "rating": [c.rating for c in checkins],
        "comment_length": [len(c.comment) for c in checkins],
        "time_of_day": [c.datetime.hour + c.datetime.minute / 60 for c in checkins],
    }

    scatters = [
        ["comment_length", "rating", {}],
        ["time_of_day", "comment_length", {}],
    ]

    if vader is not None:
        analyzer = vader.SentimentIntensityAnalyzer()
        sentiments = [analyzer.polarity_scores(c.comment) for c in checkins]
        dict_of_things["sentiment_compound"] = [s["compound"] for s in sentiments]
        dict_of_things["sentiment_pos"] = [s["pos"] for s in sentiments]
        dict_of_things["sentiment_neg"] = [s["neg"] for s in sentiments]
        scatters.extend(
            [
                ["sentiment_compound", "rating", {}],
                ["sentiment_pos", "rating", {}],
                ["sentiment_neg", "rating", {}],
            ]
        )
    else:
        warnings.warn(
            "nltk is not installed; skipping sentiment scatter plots. "
            "To enable them, install nltk: pip install nltk (and run nltk.download('vader_lexicon'))"
        )

    for x, y, kwargs in scatters:
        scatter_things(
            dict_of_things=dict_of_things,
            key_x=x,
            key_y=y,
            **kwargs,
            out_file=out_dir / f"scatter_{y}_vs_{x}.png",
        )


if __name__ == "__main__":
    CHECKINS = untappd.load_latest_checkins()
    save_various_scatters(CHECKINS)
