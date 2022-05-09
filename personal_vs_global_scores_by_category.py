import json
from pathlib import Path
import random

import pandas as pd
import seaborn
import matplotlib.pyplot as plt

data = json.loads(
    sorted(
        (Path("__file__").parent / "data_sources").glob("*.json"),
        key = lambda x: x.stat().st_mtime, reverse=True,
    )[0].read_text()
)

df = pd.DataFrame(data)

def categorize(checkin):
    typestr = checkin["beer_type"].lower()
    if "stout" in typestr:
        return "stout"
    #if "porter" in typestr:
    #    return "porter"
    if "sour" in typestr or "lambic" in typestr:
        return "sour"
    if "ipa" in typestr:
        return "ipa"
    if "lager" in typestr or "pilsner" in typestr:
        return "lager"
    if "ale" in typestr:
        return "other ale"
    return "other"


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
df["category"] = df.apply(categorize, axis=1)

JIGGLE = 0.18
df["rating_score_1"] = df.apply(lambda x: x["rating_score"] + JIGGLE * (random.random() - 0.5), axis=1)

df = df[df["category"] != "other"]

seaborn.scatterplot(
    data=df, x="global_rating_score", y="rating_score_1", hue="category",
    alpha=0.7, sizes=10,
)
# seaborn.regplot(data=df, x="global_rating_score", y="rating_score")
plt.show()
