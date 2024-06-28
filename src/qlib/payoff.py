# import numpy as np

import numpy as np

from qlib.brownian import Path


def call(paths: Path, r: float, k_strike: float, T: float):
    return np.exp(-r * T) * (paths.x[..., -1] - k_strike).clip(0)


def put(paths: Path, r: float, k_strike: float, T: float):
    return np.exp(-r * T) * (k_strike - paths.x[..., -1]).clip(0)
