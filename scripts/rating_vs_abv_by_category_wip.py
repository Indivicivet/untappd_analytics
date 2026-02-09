from collections import defaultdict
import matplotlib.pyplot as plt
import untappd
import untappd_categorise





# messy:)
C = untappd.load_latest_checkins()
cats = defaultdict(list)
for c in C: cats[untappd_categorise.festival_with_year(c)].append((c.rating, c.beer.abv))
ratings = {k: sum(a for a, _ in v)/len(v) for k, v in cats.items()}
abvs = {k: sum(b for _, b in v)/len(v) for k, v in cats.items()}
plt.scatter(abvs.values(), ratings.values())
plt.show()
