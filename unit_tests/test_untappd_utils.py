import numpy as np
import pytest

import untappd_utils


@pytest.mark.parametrize(
    "x0, y0, samples, amount, expected_x_len",
    [
        (
            np.array([0, 10]), np.array([0, 10]), 5, 0.1, 5,
        ),
        (
            np.array([-10, 0, 10]), np.array([-10, 0, 10]), 10, 0.1, 10,
        ),
    ],
)
def test_smooth_ratings_linear_increase(
    x0, y0, samples, amount, expected_x_len
):
    x_out, y_out = untappd_utils.smooth_ratings(
        x0, y0, samples=samples, amount=amount
    )
    assert len(x_out) == expected_x_len
    assert len(y_out) == expected_x_len
    assert np.all(np.diff(y_out) >= 0), (
        "y_out should be monotonically increasing"
    )


@pytest.mark.parametrize(
    "x0, y0, samples, amount, expected_x_len",
    [
        pytest.param(
            np.array([0, 5, 10]), np.array([0, 25, 100]), 100, 0.05, 100,
            id="quadratic"
        ),
        pytest.param(
            np.array([-5, 0, 5, 10]), np.array([25, 0, 25, 100]),
            50, 0.1, 50,
            id="mixed"
        ),
    ],
)
def test_smooth_ratings_non_linear(x0, y0, samples, amount, expected_x_len):
    x_out, y_out = untappd_utils.smooth_ratings(
        x0, y0, samples=samples, amount=amount
    )
    assert len(x_out) == expected_x_len
    assert len(y_out) == expected_x_len
    assert not np.allclose(np.diff(y_out, 2), 0, atol=1e-2), (
        "y_out should exhibit non-linear characteristics"
    )



def test_smooth_ratings_min_samples():
    x_out, y_out = untappd_utils.smooth_ratings(
        np.array([1, 2, 3]),
        np.array([1, 4, 9]),
        samples=2,
        amount=0.1,
    )
    assert len(x_out) == 2
    assert len(y_out) == 2


@pytest.mark.parametrize(
    "x0, y0, samples, amount, expected_x_len",
    [
        pytest.param(
            np.array([-5, 0, 5]), np.array([25, 0, 25]), 50, 0.5, 50,
            id="high_smoothing"
        ),
        pytest.param(
            np.array([-10, -5, 0, 5, 10]), np.array([100, 25, 0, 25, 100]),
            100, 0.5, 100,
            id="very_high_smoothing"
        ),
    ],
)
def test_smooth_ratings_high_smoothing(x0, y0, samples, amount, expected_x_len):
    x_out, y_out = untappd_utils.smooth_ratings(
        x0, y0, samples=samples, amount=amount
    )
    assert len(x_out) == expected_x_len
    assert len(y_out) == expected_x_len
    assert np.var(y_out) < np.var(y0) * 0.8, (
        "y_out variance should be less than y0 variance due to smoothing"
    )
