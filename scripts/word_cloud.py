import statistics
import string
from collections import defaultdict

from matplotlib import pyplot as plt
from wordcloud import WordCloud

import untappd

CHECKINS = untappd.load_latest_checkins()

word_ratings = defaultdict(list)
all_ratings = []

for c in CHECKINS:
    all_ratings.append(c.rating)
    for word in c.comment.split():
        word_alpha = "".join(
            c for c in word.lower() if c in string.ascii_lowercase
        )
        word_ratings[word_alpha].append(c.rating)

mean = statistics.mean(all_ratings)
std = statistics.stdev(all_ratings)

positivity = {
    word: statistics.mean(
        [max(rating - mean, 0) for rating in ratings]
    ) / std
    for word, ratings in word_ratings.items()
}
negativity = {
    word: statistics.mean(
        [max(mean - rating, 0) for rating in ratings]
    ) / std
    for word, ratings in word_ratings.items()
}
colors = {
    word: (
        positivity,

    )
    for word in word_ratings
}
occurrences = {word: len(ratings) for word, ratings in word_ratings.items()}

print(word_ratings)
wc = WordCloud().generate_from_frequencies(occurrences)

plt.imshow(wc)
plt.show()
