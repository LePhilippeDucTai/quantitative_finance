from typing import Any, Callable

import numba
import numpy as np

from qlib.brownian import Path, TimeGrid
from qlib.constant_parameters import DEFAULT_RNG, N_DYADIC
from qlib.numerical.euler_scheme import euler_discretization
from qlib.optimized_numba import euler_discretization_jit
from qlib.utils.misc import to_tuple
from qlib.utils.timing import time_it


class ItoProcess:
    """Base class for Ito Processes.

    Implements generic Monte-Carlo Euler with numba's just-in-time compilation

    Diffusion :
    dXt = mu(t, Xt)dt + sigma(t, Xt)dWt
    This class gives an interface to define general Ito Processes
    by giving the mu and sigma functions.
    """

    def __init__(self):
        self.model_args = ()

    def mu(self, t: float, xt: float):
        return 0

    def sigma(self, t: float, xt: float):
        return 0


class EulerSchema(ItoProcess):

    def mu(self, t, x):
        return

    def sigma(self, t, x):
        return

    @time_it
    def mc_euler(
        self,
        x0: float,
        time_horizon: float,
        size: int | tuple[int],
        n_dyadic: int = N_DYADIC,
        generator: np.random.Generator = DEFAULT_RNG,
    ):
        time_grid = TimeGrid(time_horizon, n_dyadic)
        dt, n_t, t = time_grid.dt, time_grid.n_dates, time_grid.t
        size = to_tuple(size)
        g = generator.normal(scale=np.sqrt(dt), size=(*size, n_t))
        xt = np.empty_like(g)
        args = self.model_args
        xt[..., 0] = x0
        xt = euler_discretization(self.mu(), self.sigma(), t, xt, n_t, g, dt, *args)
        return Path(t, xt)


class EulerSchemaJit(ItoProcess):

    def mu_jit(self) -> Callable[[float, float, Any], float]:
        return numba.njit(self.mu())

    def sigma_jit(self) -> Callable[[float, float, Any], float]:
        return numba.njit(self.sigma())

    @time_it
    def mc_euler_jit(
        self,
        x0: float,
        time_horizon: float,
        size: tuple[int],
        n_dyadic: int = N_DYADIC,
        generator: np.random.Generator = DEFAULT_RNG,
    ) -> Path:
        time_grid = TimeGrid(time_horizon, n_dyadic)
        dt, n_t, t = time_grid.dt, time_grid.n_dates, time_grid.t
        size = to_tuple(size)
        g = generator.normal(scale=np.sqrt(dt), size=(*size, n_t))
        xt = np.empty_like(g)
        args = self.model_args
        xt[..., 0] = x0
        xt = euler_discretization_jit(
            self.mu_jit(), self.sigma_jit(), t, xt, n_t, g, dt, *args
        )
        return Path(t, xt)
