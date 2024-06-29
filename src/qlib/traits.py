from typing import Any, Callable

import numba
import numpy as np
from loguru import logger

from qlib.constant_parameters import DEFAULT_RNG, N_DYADIC
from qlib.models.path import Path, TimeGrid
from qlib.numerical.euler_scheme import euler_discretization
from qlib.numerical.optimized_numba import euler_discretization_jit
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

    def __init__(self, x0: float, time_horizon: float):
        self.model_args = ()
        self.x0 = x0
        self.time_horizon = time_horizon

    def mu(self, t: float, xt: float):
        return 0

    def sigma(self, t: float, xt: float):
        return 0

    def initialize_discretization(
        self, size: int | tuple, n_dyadic=N_DYADIC, generator=DEFAULT_RNG
    ):
        time_grid = TimeGrid(self.time_horizon, n_dyadic)
        dt, n_t, t = time_grid.dt, time_grid.n_dates, time_grid.t
        size = to_tuple(size)
        g = generator.normal(scale=np.sqrt(dt), size=(*size, n_t))
        xt = np.empty_like(g)
        xt[..., 0] = self.x0
        return dt, n_t, t, g, xt


class EulerSchema(ItoProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mu(self, t, x):
        logger.warning("Not implemented yet !")
        return

    def sigma(self, t, x):
        logger.warning("Not implemented yet !")
        return

    @time_it
    def mc_euler(
        self,
        size: int | tuple[int],
        n_dyadic: int = N_DYADIC,
        generator: np.random.Generator = DEFAULT_RNG,
    ):
        dt, n_t, t, g, xt = self.initialize_discretization(size, n_dyadic, generator)
        xt = euler_discretization(
            self.mu(), self.sigma(), t, xt, n_t, g, dt, *self.model_args
        )
        return Path(t, xt)


class EulerSchemaJit(ItoProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mu_jit(self) -> Callable[[float, float, Any], float]:
        return numba.njit(self.mu())

    def sigma_jit(self) -> Callable[[float, float, Any], float]:
        return numba.njit(self.sigma())

    @time_it
    def mc_euler_jit(
        self,
        size: tuple[int],
        n_dyadic: int = N_DYADIC,
        generator: np.random.Generator = DEFAULT_RNG,
    ) -> Path:
        dt, n_t, t, g, xt = self.initialize_discretization(size, n_dyadic, generator)
        xt = euler_discretization_jit(
            self.mu_jit(), self.sigma_jit(), t, xt, n_t, g, dt, *self.model_args
        )
        return Path(t, xt)


class Model(EulerSchema, EulerSchemaJit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mc_exact(
        self,
        size: tuple[int],
        n_dyadic: int = N_DYADIC,
        generator: np.random.Generator = DEFAULT_RNG,
    ) -> Path:
        logger.warning("Not implemented yet!")
        return
