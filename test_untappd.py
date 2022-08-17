import json

import untappd


BASIC_JSON_SINGLE_BEER_EXAMPLE_1 = json.loads('''
{
"beer_name":"Beer Nameee",
"brewery_name":"Brewery Nameee",
"beer_type":"it's a beer",
"beer_abv":"4.2",
"beer_ibu":"42",
"comment":"",
"venue_name":null,
"venue_city":null,
"venue_state":null,
"venue_country":null,
"venue_lat":null,
"venue_lng":null,
"rating_score":"3.5",
"created_at":"2002-02-20 11:22:33",
"checkin_url":"https:\/\/untappd.com\/c\/25232",
"beer_url":"https:\/\/untappd.com\/beer\/345123",
"brewery_url":"https:\/\/untappd.com\/brewery\/345345",
"brewery_country":"England",
"brewery_city":"London",
"brewery_state":"London",
"flavor_profiles":"",
"purchase_venue":"",
"serving_type":"",
"checkin_id":"123235325",
"bid":"2352354",
"brewery_id":"23423",
"photo_url":null,
"global_rating_score":1.23,
"global_weighted_rating_score":3.45,
"tagged_friends":"",
"total_toasts":"0",
"total_comments":"0"
}
''')


def test_from_dict_runs():
    untappd.Checkin.from_dict(BASIC_JSON_SINGLE_BEER_EXAMPLE_1)


def test_Checkin_from_dict_to_dict_consistent():
    """
    dict -> Checkin -> dict
    """
    source_dict = BASIC_JSON_SINGLE_BEER_EXAMPLE_1
    checkin = untappd.Checkin.from_dict(source_dict)
    result = checkin.to_dict()
    mismatches = []
    for field, expected in source_dict.items():
        if expected is None or expected == "":
            continue  # don't worry about missing fields
        if (got := result.get(field, "MISSING!!!!")) != expected:
            mismatches.append((field, expected, got))
    assert not mismatches, f"""mismatches (field, expected, got):
{mismatches}"""


def test_Checkin_to_dict_from_dict_consistent_1():
    """
    Checkin -> dict -> Checkin
    """
    checkin = untappd.Checkin(
        beer=untappd.Beer(
            name="my beerly",
            abv=3.5,
            ibu=101,
            id=99999,
            brewery=untappd.Brewery(
                name="excellent brewERY",
            ),
        ),
    )
    resulting_dict = checkin.to_dict()
    resulting_checkin = untappd.Checkin.from_dict(resulting_dict)
    assert resulting_checkin == checkin
