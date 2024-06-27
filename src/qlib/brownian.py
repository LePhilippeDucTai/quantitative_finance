import functools as ft
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from qlib.utils.misc import to_tuple
from qlib.utils.timing import time_it

DEFAULT_GENERATOR = np.random.default_rng()


@dataclass
class Path:
    """Path class."""

    t: np.ndarray
    x: np.ndarray

    def plot(self):
        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        for path in self.x:
            ax.plot(self.t, path)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        plt.show()
        return ax


@dataclass(frozen=True)
class TimeGrid:
    maturity: float
    n_dyadic: int

    @ft.cached_property
    def _d(self):
        return 2**self.n_dyadic

    @ft.cached_property
    def t(self):
        return np.linspace(0, self.maturity, self.n_dates)

    @ft.cached_property
    def n_dates(self):
        return int(self.maturity * self._d)

    @ft.cached_property
    def dt(self):
        return 1 / (self._d - 1)


def plot_brownians(p: Path) -> None:
    """Plot the path."""
    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    for path in p.x:
        ax.plot(p.t, path)
    ax.set_title("Standard Brownian Motion sample paths")
    ax.set_xlabel("Time")
    ax.set_ylabel("Asset Value")
    ic = 1.96 * np.sqrt(p.t)
    ax.plot(p.t, ic, label="sqrt(t)", color="black", linestyle="--")
    ax.plot(p.t, -ic, color="black", linestyle="--")
    ax.legend()
    plt.show()


@time_it
def brownian_trajectories(
    t: float,
    size: int | tuple = 1,
    n: int = 6,
    gen: np.random.Generator = DEFAULT_GENERATOR,
) -> Path:
    """Generate Brownian trajectories.
    Size can be a tuple, easing simulating several trajectories.
    """
    size = to_tuple(size)
    time_grid = TimeGrid(t, n)
    dt, n_t = time_grid.dt, time_grid.n_dates
    y = gen.normal(scale=np.sqrt(dt), size=(*size, n_t))
    y[..., 0] = 0
    return Path(time_grid.t, y.cumsum(axis=-1))
