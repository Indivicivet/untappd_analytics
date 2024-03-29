import statistics
import string
from collections import defaultdict

import wordcloud

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
        if (
            word_alpha in wordcloud.STOPWORDS
            or word_alpha in {
                "checkin", "posthoc", "drank", "back", "think", "beer",
            }
        ):
            continue
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

COLOR_MAG = 196
colors = {
    word: (
        int(COLOR_MAG * negativity[word]),
        int(COLOR_MAG * positivity[word]),
        COLOR_MAG - int(COLOR_MAG / 2 * (positivity[word] + negativity[word])),
    )
    for word in word_ratings
}
occurrences = {
    word: len(ratings)
    for word, ratings in word_ratings.items()
}

print(word_ratings)
wc = wordcloud.WordCloud(
    width=1280,
    height=720,
    max_words=1000,
    max_font_size=200,
    prefer_horizontal=0.95,
).generate_from_frequencies(occurrences).recolor(
    color_func=(lambda word, **kwargs: colors[word]),
)

wc.to_image().show()
