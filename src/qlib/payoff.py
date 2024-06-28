# import numpy as np

import numpy as np

from qlib.brownian import Path


def european_call(strike: float) -> callable:
    def aux(paths: Path):
        return (paths.x[..., -1] - strike).clip(0)

    return aux


def european_put(paths: Path, strike: float) -> callable:
    return (strike - paths.x[..., -1]).clip(0)
