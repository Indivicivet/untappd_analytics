from collections import defaultdict
import matplotlib.pyplot as plt
import untappd


# copy-pasted from rating_histogram_by_various
# todo: move all the categorising functions to untappd_categorisers or w/e?
def festival_with_year(checkin, include_non_festival=True):
    if (
        checkin.venue is not None
        and "beer festival" in checkin.venue.name.lower()
    ):
        return f"{checkin.venue.name} ({checkin.datetime.year})"
    if include_non_festival:
        return "Non-festival"
    return None


# messy:)
C = untappd.load_latest_checkins()
cats = defaultdict(list)
for c in C: cats[festival_with_year(c)].append((c.rating, c.beer.abv))
ratings = {k: sum(a for a, _ in v)/len(v) for k, v in cats.items()}
abvs = {k: sum(b for _, b in v)/len(v) for k, v in cats.items()}
plt.scatter(abvs.values(), ratings.values())
plt.show()
