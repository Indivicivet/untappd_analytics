import pytest
import untappd


def make_checkin(beer_name="Beer", rating=3.5, brewery="Brewery", abv=5.0):
    return untappd.Checkin(
        beer=untappd.Beer(
            name=beer_name,
            abv=abv,
            brewery=untappd.Brewery(name=brewery),
        ),
        rating=rating,
    )


def test_magic_rating_empty():
    score, beers = untappd.magic_rating([])
    assert score == 0
    assert beers == []


def test_magic_rating_dropoff_1_gives_neutral():
    # When dropoff_ratio=1.0, (1 - dropoff_ratio) * sum(...) = 0
    # So the second term becomes (1 - average_score_weight) * neutral_rating
    checkins = [
        make_checkin("A", 4.0),
        make_checkin("B", 3.0),
    ]
    # Simple average is 3.5
    # (0.35 * 3.5) + (1 - 0.35) * 3.25 = 1.225 + 2.1125 = 3.3375
    score, _ = untappd.magic_rating(checkins, dropoff_ratio=1.0)
    assert score == pytest.approx(3.3375)


def test_magic_rating_pure_average_weight_1():
    checkins = [
        make_checkin("A", 4.0),
        make_checkin("B", 3.0),
    ]
    # Simple average is 3.5
    score, _ = untappd.magic_rating(checkins, average_score_weight=1.0)
    assert score == pytest.approx(3.5)


def test_magic_rating_neutral_impact():
    neutral = 3.5
    checkins = [
        make_checkin("A", neutral),
        make_checkin("B", neutral),
        make_checkin("C", neutral),
    ]
    score, _ = untappd.magic_rating(checkins, neutral_rating=neutral)
    assert score == pytest.approx(neutral)


def test_magic_rating_additive_score():
    # If average_score_weight=0 and neutral_rating=0, score should be additive
    # (well, scaled by dropoff)
    checkins_1 = [make_checkin("A", 4.0)]
    checkins_2 = [make_checkin("A", 4.0), make_checkin("B", 4.0)]

    score1, _ = untappd.magic_rating(
        checkins_1, average_score_weight=0, neutral_rating=0, dropoff_ratio=0.5
    )
    score2, _ = untappd.magic_rating(
        checkins_2, average_score_weight=0, neutral_rating=0, dropoff_ratio=0.5
    )

    # 4.0 * (1-0.5) * (0.5^0) = 2.0
    assert score1 == pytest.approx(2.0)
    # 2.0 + 4.0 * (1-0.5) * (0.5^1) = 2.0 + 4.0 * 0.5 * 0.5 = 3.0
    assert score2 == pytest.approx(3.0)
    assert score2 > score1


def test_magic_rating_single_neutral():
    score, _ = untappd.magic_rating([make_checkin("A", 3.25)], neutral_rating=3.25)
    assert score == pytest.approx(3.25)
