import functools as ft
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


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
    """Defines a time grid for Euler Discretization Schemes."""

    maturity: float
    n_dyadic: int

    @ft.cached_property
    def _d(self) -> int:
        return 2**self.n_dyadic

    @ft.cached_property
    def t(self) -> np.ndarray:
        return np.linspace(0, self.maturity, self.n_dates)

    @ft.cached_property
    def n_dates(self) -> int:
        return int(self.maturity * self._d)

    @ft.cached_property
    def dt(self) -> float:
        return 1 / (self._d - 1)
