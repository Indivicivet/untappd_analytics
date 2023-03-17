import random
from collections import Counter

import pandas as pd
import seaborn
import matplotlib.pyplot as plt

import untappd

data = untappd.load_latest_datafile()

df = pd.DataFrame(data)

seaborn.set()


def abv_categorize(checkin):
    abv = checkin["beer_abv"]
    try:
        abv = float(abv)
    except:
        return "unknown"
    if abv <= 3.5:
        return "weak"
    if abv <= 6:
        return "standard"
    #if abv <= 9.5:
    return "strong"


TOP_VENUES = [
    venue
    for venue, _ in Counter(
        untappd.Checkin.from_dict(d).venue
        for d in data
        if d.get("venue_name")
    ).most_common(2)
]


def by_top_venue(d):
    v = untappd.Checkin.from_dict(d).venue
    if v in TOP_VENUES:
        return v.name
    return "other"


df = df[df["rating_score"] != ""]
df["rating_score"] = df["rating_score"].astype(float)
df["beer_abv_category"] = df.apply(abv_categorize, axis=1)
df["category"] = df.apply(
    # lambda d: untappd.Beer.from_checkin_dict(d).get_style_category(),
    by_top_venue,
    axis=1,
)

JIGGLE = 0.18
df["rating_score_1"] = df.apply(lambda x: x["rating_score"] + JIGGLE * (random.random() - 0.5), axis=1)

df = df[df["category"] != "other"]


def personal_vs_global():  # globals yay!!
    global df
    df = df[df["global_rating_score"].astype(float) > 0]
    seaborn.scatterplot(
        data=df, x="global_rating_score", y="rating_score_1", hue="category",
        alpha=0.7, sizes=10,
    )


def score_vs_abv():
    global df
    df["beer_abv"] = df["beer_abv"].astype(float)
    df = df.sort_values("beer_abv")
    seaborn.scatterplot(
        data=df, x="beer_abv", y="rating_score_1", hue="category",
        alpha=0.2, sizes=10,
    )


# seaborn.regplot(data=df, x="global_rating_score", y="rating_score")

plt.figure(figsize=(10, 10))
personal_vs_global()
# score_vs_abv()
plt.show()
