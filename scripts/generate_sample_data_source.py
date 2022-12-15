import datetime
import json
import random

import untappd


# todo :: scripts /actually/ assume lots of optional data is present
# so this may not be sufficient yet to generate suitable output


# todo :: make untappd.load_latest_checkins() ignore samples
# if other files are present


def make_random_brewery():
    return untappd.Brewery(
        name=(
            random.choice("hello good London".split())
            + random.choice("brewery beers brewers".split())
        ),
    )


def make_random_beer():
    return untappd.Beer(
        name=f"yummy beer #{random.randint(1, 99999999)}",
        abv=random.random() * 15,
        ibu=random.randint(0, 100),
        brewery=make_random_brewery(),
        type=random.choice("IPA,Stout,Pale Ale".split(",")),
    )


def make_random_checkin():
    return untappd.Checkin(
        beer=make_random_beer(),
        rating=int(random.gauss(3.2, 1) * 4) / 4,
        datetime=datetime.datetime(
            2020,
            random.randint(1, 12),
            random.randint(1, 28),
            random.randrange(24),
            random.randrange(60),
            random.randrange(60),
        ),
    )


seed = 123
random.seed(seed)
checkin_dicts = [
    make_random_checkin().to_dict()
    for _ in range(1000)
]
assert not target_file.exists()  # comment out if happy overwriting
target_file = untappd.DEFAULT_DATA_SOURCE / f"sample_data_{seed=}.json"
target_file.write_text(json.dumps(checkin_dicts))
print(f"successfully wrote out to {target_file}")
