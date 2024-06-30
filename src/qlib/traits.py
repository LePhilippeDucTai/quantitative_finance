from dataclasses import dataclass
from typing import Any, Callable

import numba
import numpy as np
from loguru import logger

from qlib.constant_parameters import DEFAULT_RNG, N_DYADIC, N_MC
from qlib.models.path import Path, TimeGrid
from qlib.numerical.optimized_numba import euler_discretization_jit
from qlib.utils.misc import to_tuple
from qlib.utils.timing import time_it


@dataclass
class ItoProcessParameters:
    x0: float


class ItoProcess:
    """Base class for Ito Processes.

    Implements generic Monte-Carlo Euler with numba's just-in-time compilation

    Diffusion :
    dXt = mu(t, Xt)dt + sigma(t, Xt)dWt
    This class gives an interface to define general Ito Processes
    by giving the mu and sigma functions.
    """

    def __init__(self, model_parameters: ItoProcessParameters):
        self.model_parameters = model_parameters

    def mu(self, t: float, xt: float):
        return 0

    def sigma(self, t: float, xt: float):
        return 0

    def initialize_discretization(
        self,
        time_horizon: int,
        size: int | tuple,
        n_dyadic=N_DYADIC,
        generator=DEFAULT_RNG,
    ):
        time_grid = TimeGrid(time_horizon, n_dyadic)
        dt, n_t, t = time_grid.dt, time_grid.n_dates, time_grid.t
        size = to_tuple(size)
        g = generator.normal(scale=np.sqrt(dt), size=(*size, n_t))
        xt = np.empty_like(g)
        xt[..., 0] = self.model_parameters.x0
        return dt, n_t, t, g, xt


def euler_discretization(  # noqa: PLR0913
    mu: callable,
    sigma: callable,
    t: np.ndarray,
    xt: np.ndarray,
    n_t: int,
    g: np.ndarray,
    dt: float,
) -> np.ndarray:
    for i in range(n_t - 1):
        t_i, x_i = t[i], xt[..., i]
        g_i = g[..., i]
        μ, σ = mu(t_i, x_i), sigma(t_i, x_i)
        xt[..., i + 1] = x_i + μ * dt + σ * g_i
    return xt


class EulerSchema(ItoProcess):
    def __init__(self, model_parameters: ItoProcessParameters, *args, **kwargs):
        super().__init__(model_parameters, *args, **kwargs)

    def mu(self, t, x):
        logger.warning("Not implemented yet !")
        return

    def sigma(self, t, x):
        logger.warning("Not implemented yet !")
        return

    @time_it
    def mc_euler(
        self,
        time_horizon: int,
        size: int | tuple[int] = N_MC,
        n_dyadic: int = N_DYADIC,
        generator: np.random.Generator = DEFAULT_RNG,
    ) -> Path:
        dt, n_t, t, g, xt = self.initialize_discretization(
            time_horizon, size, n_dyadic, generator
        )
        xt = euler_discretization(self.mu, self.sigma, t, xt, n_t, g, dt)
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
            self.mu_jit(), self.sigma_jit(), t, xt, n_t, g, dt, self.model_parameters
        )
        return Path(t, xt)


class InterestRatesModel:

    def instantaneous(self, size: int) -> np.ndarray | float:
        pass

    def kind(self) -> str:
        pass


class TermStructure:
    def __init__(self, rates_model: InterestRatesModel):
        self.rates_model = rates_model

    def discount_factor(self, time_to_maturity: float) -> float | np.ndarray:
        if self.rates_model.kind() == "flat":
            rate = self.rates_model.instantaneous() * time_to_maturity
        else:
            raise Exception("Not implemented yet.")
        return np.exp(-rate)


@dataclass
class FlatForward(InterestRatesModel):
    risk_free_rate: float

    def instantaneous(self, _: int = 0) -> float:
        return self.risk_free_rate

    def kind(self):
        return "flat"


class Model(EulerSchema, EulerSchemaJit):
    def __init__(
        self,
        model_parameters: ItoProcessParameters,
        term_structure: TermStructure,
        *args,
        **kwargs
    ):
        super().__init__(model_parameters, *args, **kwargs)
        self.term_structure = term_structure

    def mc_exact(
        self,
        size: tuple[int],
        n_dyadic: int = N_DYADIC,
        generator: np.random.Generator = DEFAULT_RNG,
    ) -> Path:
        logger.warning("Not implemented yet!")
        return

    def mc_terminal(
        self,
        time_horizon: float,
        size: tuple[int],
        generator: np.random.Generator = DEFAULT_RNG,
    ) -> Path:
        logger.warning("Not implemented yet!")
        return
