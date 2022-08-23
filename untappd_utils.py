
import numpy as np
from scipy import interpolate, ndimage


def smooth_ratings(x0, y0, samples=200):
    x_smooth = np.linspace(min(x0), max(x0), samples, endpoint=True)
    y_linear = interpolate.interp1d(x0, y0)(x_smooth)
    return x_smooth, ndimage.gaussian_filter(y_linear, samples * 0.01)
