import untappd

CHECKINS = untappd.load_latest_checkins()

for ci1, ci2 in zip(CHECKINS, CHECKINS[1:]):
    if (
        (ci1.datetime - ci2.datetime).total_seconds() < 600
        and ci1.beer == ci2.beer
        # and ci1.comment == ci2.comment
    ):
        print(ci1.beer)
        print(ci1.datetime, ci1.comment)
        print(ci2.datetime, ci2.comment)
