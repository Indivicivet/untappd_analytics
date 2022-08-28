import random

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


df = df[df["rating_score"] != ""]
df["rating_score"] = df["rating_score"].astype(float)
df["beer_abv_category"] = df.apply(abv_categorize, axis=1)
df["category"] = df.apply(
    lambda d: untappd.Beer.from_checkin_dict(d).get_style_category(),
    axis=1,
)

JIGGLE = 0.18
df["rating_score_1"] = df.apply(lambda x: x["rating_score"] + JIGGLE * (random.random() - 0.5), axis=1)

df = df[df["category"] != "other"]

# personal vs global by category:
# seaborn.scatterplot(
#     data=df, x="global_rating_score", y="rating_score_1", hue="category",
#     alpha=0.7, sizes=10,
# )

# score vs ABV by category:
df["beer_abv"] = df["beer_abv"].astype(float)
df = df.sort_values("beer_abv")
seaborn.scatterplot(
    data=df, x="beer_abv", y="rating_score_1", hue="category",
    alpha=0.2, sizes=10,
)
# seaborn.regplot(data=df, x="global_rating_score", y="rating_score")
plt.show()
