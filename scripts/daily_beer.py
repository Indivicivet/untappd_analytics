from datetime import datetime

import untappd

cis = untappd.load_latest_checkins()

recent = sorted(
    [
        c
        for c in cis
        if c.datetime >= datetime(year=2024, month=1, day=1)
            and c.beer.get_style_category() == "ipa"
            and c.beer.abv < 9
    ],
    key=lambda c: c.rating,
    reverse=True,
)
for i, r in enumerate(recent[:20]):
    print(f"#{i+1}: {r.beer}, {r.rating}")
