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


def test_to_dict_consistent():
    checkin = untappd.Checkin.from_dict(BASIC_JSON_SINGLE_BEER_EXAMPLE_1)
    result = checkin.to_dict()
