import functools
import statistics
from pathlib import Path
from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate, ndimage

import untappd


def smooth_ratings(x0, y0, samples=200, amount=0.1):
    x_smooth = np.linspace(min(x0), max(x0), samples, endpoint=True)
    y_linear = interpolate.interp1d(x0, y0)(x_smooth)
    y_smooth = ndimage.gaussian_filter(
        y_linear,
        samples * amount / (max(x0) - min(x0)),
    )
    return x_smooth, y_smooth


def show_or_save_to_out_file(func):
    @functools.wraps(func)
    def wrapped(
        *args,
        out_file: Optional[Path] = None,
        **kwargs,
    ):
        result = func(*args, **kwargs)
        if out_file is None:
            plt.show()
        else:
            out_file = Path(out_file)
            out_file.parent.mkdir(exist_ok=True, parents=True)
            print(f"saved to {out_file}")
            plt.savefig(out_file, bbox_inches="tight")
        return result  # probably None, if it's just plotting things.
    return wrapped


def mean_and_std(
    checkins_or_values: list[Union[untappd.Checkin, float]]
) -> tuple[float, float]:
    all_ratings = (
        [ci.rating for ci in checkins_or_values]
        if isinstance(checkins_or_values[0], untappd.Checkin)
        else checkins_or_values
    )
    return statistics.mean(all_ratings), statistics.stdev(all_ratings)


def mean_plus_minus_std(
    checkins_or_values: list[Union[untappd.Checkin, float]]
) -> tuple[float, float, float]:
    mean, std = mean_and_std(checkins_or_values)
    return mean - std, mean, mean + std
