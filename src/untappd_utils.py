
import numpy as np
from scipy import interpolate, ndimage


def smooth_ratings(x0, y0, samples=200, amount=0.1):
    x_smooth = np.linspace(min(x0), max(x0), samples, endpoint=True)
    y_linear = interpolate.interp1d(x0, y0)(x_smooth)
    y_smooth = ndimage.gaussian_filter(
        y_linear,
        samples * amount / (max(x0) - min(x0)),
    )
    return x_smooth, y_smooth
