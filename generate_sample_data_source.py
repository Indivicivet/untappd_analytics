import json

import untappd


print("THIS FILE DOES NOT GIVE A SENSIBLE SAMPLE DATA SOURCE YET")
print("SO DO NOT USE IT. (or modify it to add all needed fields...)")


# todo :: all of the scripts /actually/ assume all data is present
# so this isn't sufficient yet to generate suitable output

# todo :: make untappd.load_latest_checkins() ignore samples
# if other files are present


def make_random_brewery():
    return untappd.Brewery(
        name="hello brewery",
    )


def make_random_beer():
    return untappd.Beer(
        name="its a beer",
        abv=5,
        ibu=50,
        brewery=make_random_brewery(),
    )


def make_random_checkin():
    return untappd.Checkin(
        beer=make_random_beer(),
    )


checkin_dicts = [
    make_random_checkin().to_dict()
    for _ in range(1000)
]
target_file = untappd.DEFAULT_DATA_SOURCE / "sample_data_123.json"
assert not target_file.exists()  # comment out if happy overwriting
target_file.write_text(json.dumps(checkin_dicts))
print(f"successfully wrote out to {target_file}")
