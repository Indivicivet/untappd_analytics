import statistics

import matplotlib.pyplot as plt

import untappd

CHECKINS = untappd.load_latest_checkins()

START = min(c.datetime for c in CHECKINS)
END = max(c.datetime for c in CHECKINS)
by_year = {
    cat: {
        year: statistics.mean([
            c.rating
            for c in CHECKINS
            if c.datetime.year == year and c.beer.get_style_category() == cat
        ])
        for year in range(START.year, END.year + 1)
    }
    for cat in untappd.CATEGORY_KEYWORDS
}

plt.figure(figsize=(12.8, 7.2))
for year, checkins_by_cat in by_year.items():
    plt.bar(
        [f"{year} {cat}" for cat in checkins_by_cat.keys()],
        checkins_by_cat.values(),
    )
plt.xticks(rotation=-45)

all_vals = [v for year in by_year.values() for v in year.values()]
plt.ylim([min(all_vals) - 0.25, max(all_vals)])
plt.show()
