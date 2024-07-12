"""Brownian class."""

import matplotlib.pyplot as plt
import numpy as np
from qlib.models.path import Path, TimeGrid
from qlib.utils.misc import to_tuple
from qlib.utils.timing import time_it
from scipy import interpolate

DEFAULT_GENERATOR = np.random.default_rng()


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


def brownian_bridge_trajectories(
    t: float, size: int | tuple = 1, n=6, gen=DEFAULT_GENERATOR
) -> Path:
    standard_brownian = brownian_trajectories(t, size, n, gen)
    k, w = standard_brownian.t, standard_brownian.x
    w_final = w[..., -1].reshape(-1, 1)
    trajectories = w - (k / t) * w_final
    return Path(k, trajectories)


def brownian_bridge_enrich(path: Path, gen=DEFAULT_GENERATOR):
    t, x = path.t, path.x
    n_points = len(t) - 1
    dt = 1 / n_points
    t_max = t[-1]
    enriched_t = np.linspace(0, t_max, 2 * n_points + 1)
    noise = gen.normal(size=x.shape, scale=np.sqrt(t_max * dt * 0.5))[:, 1:]
    f = interpolate.interp1d(t, x, axis=-1)
    result = f(enriched_t)
    result[:, 1::2] += noise
    return Path(enriched_t, result)


def brownian_trajectories_exact(
    t: float, size: int | tuple, n=6, gen=DEFAULT_GENERATOR
):
    size = to_tuple(size)
    i = 0
    x0 = gen.normal(scale=np.sqrt(t), size=(*size, 2))
    x0[..., 0] = 0.0
    path = Path(np.linspace(0, t, 2), x0)
    while i < n:
        path = brownian_bridge_enrich(path, gen)
        i += 1
    return path


if __name__ == "__main__":
    t, size = 25, 100
    gen = np.random.default_rng(113121)
    paths = brownian_trajectories_exact(t, size, n=8, gen=gen)
    # paths.plot()
    plot_brownians(paths)
