"""Main module."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from qlib.utils import to_tuple

DEFAULT_GENERATOR = np.random.default_rng()


@dataclass
class Path:
    """Path class."""

    t: np.ndarray
    x: np.ndarray


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


def brownian_trajectories(
    t: int,
    size: int | tuple = 1,
    n: int = 6,
    gen: np.random.Generator = DEFAULT_GENERATOR,
) -> Path:
    """Generate a Brownian path."""
    size = to_tuple(size)
    n_t = t * 2**n
    x = np.linspace(0, t, n_t)
    sigma = np.sqrt(1 / (2**n - 1))
    y = gen.normal(scale=sigma, size=(*size, n_t))
    y[..., 0] = 0
    return Path(x, y.cumsum(axis=-1))


def main() -> None:
    """Test."""
    p = brownian_trajectories(1, size=100, n=10)
    plot_brownians(p)


if __name__ == "__main__":
    main()
