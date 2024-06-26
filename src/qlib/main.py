"""Main module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class Path:
    """Path class."""

    t: np.ndarray
    x: np.ndarray


def to_tuple(t: int | tuple) -> tuple:
    """Convert t to a tuple."""
    if isinstance(t, int):
        return (t,)
    return t


def brownian_trajectories(t: int, n: int, size: int | tuple = 1) -> Path:
    """Generate a Brownian path."""
    size = to_tuple(size)
    n_t = t * 2**n
    x = np.linspace(0, t, n_t)
    y = np.random.default_rng().normal(scale=1 / n_t, size=(*size, n_t))
    y[..., 0] = 0
    return Path(x, y.cumsum(axis=-1))


def main() -> None:
    """Test."""
    p = brownian_trajectories(1, 2, size=1)
    logger.info(p)
    assert len(p.t) == len(p.x)


if __name__ == "__main__":
    main()
