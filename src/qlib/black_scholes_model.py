"""Black-Scholes model for pricing options."""

import functools as ft
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numba
import numpy as np
from scipy.special import ndtr

import qlib.payoff as payoff
from qlib.brownian import Path, TimeGrid, brownian_trajectories
from qlib.term_structure import DayCountConvention, FlatForward, FlatTermStructure
from qlib.utils.logger import logger
from qlib.utils.misc import to_tuple
from qlib.utils.timing import time_it

# Gives by default for each time interval [t, t + 1], 2 ** 10 = 1024 points
N_DYADIC = 4


@numba.njit
def euler_discretization(
    mu_jit: callable,
    sigma_jit: callable,
    t: np.ndarray,
    xt: np.ndarray,
    n_t: int,
    g: np.ndarray,
    dt: float,
) -> np.ndarray:
    for i in range(n_t - 1):
        t_i, x_i = t[i], xt[..., i]
        g_i = g[..., i]
        μ, σ = mu_jit(t_i, x_i), sigma_jit(t_i, x_i)
        xt[..., i + 1] = x_i + μ * dt + σ * g_i
    return xt


@numba.njit
def bs_mu_jit(r, _, x):
    return r * x


@numba.njit
def bs_sigma_jit(sig, _, x):
    return sig * x


@numba.njit
def dummy(t, x):
    return t * x


# Compiling the functions
_ = bs_mu_jit(0, 0, 0)
_ = bs_sigma_jit(0, 0, 0)
_ = euler_discretization(dummy, dummy, np.arange(1), np.arange(1), 1, np.arange(1), 1)


@dataclass
class ModelParameters(ABC):

    @abstractmethod
    def discount_factor(self, *args, **kwargs) -> float:
        pass


@dataclass
class BlackScholesParameters(ModelParameters):
    """Parameters for the Black-Scholes model."""

    r: float
    sig: float


class ItoProcess:
    """Base class for Ito Processes.

    Implements generic Monte-Carlo Euler with numba's just-in-time compilation

    Diffusion :
    dXt = mu(t, Xt)dt + sigma(t, Xt)dWt
    This class gives an interface to define general Ito Processes
    by giving the mu and sigma functions.
    """

    def __init__(self, term_structure: FlatTermStructure, s0: float):
        self.term_structure = term_structure
        self.s0 = s0

    def mu_jit(self):
        return 0

    def sigma_jit(self):
        return 0

    @time_it
    def mc_euler(
        self,
        maturity: float,
        size: tuple[int],
        n_dyadic: int = N_DYADIC,
    ) -> Path:
        time_grid = TimeGrid(maturity, n_dyadic)
        dt, n_t, t = time_grid.dt, time_grid.n_dates, time_grid.t
        size = to_tuple(size)
        print(size)
        g = np.random.default_rng().normal(scale=np.sqrt(dt), size=(*size, n_t))
        xt = np.empty_like(g)

        mu_jit = self.mu_jit
        sigma_jit = self.sigma_jit
        xt[..., 0] = self.s0
        xt = euler_discretization(mu_jit, sigma_jit, t, xt, n_t, g, dt)
        return Path(t, xt)


class BlackScholesModelDeterministic:
    """Black-Scholes model for pricing options."""

    def __init__(self, ts: FlatTermStructure, s0: float, sigma: float) -> None:
        """Initialize the model with a BlackScholesModelParameters object."""
        self.term_structure = ts
        self.s0 = s0
        self.sigma = sigma

    def call(self, time_to_maturity: float, k: float) -> float:
        """Calculate the price of a call option."""
        t = time_to_maturity
        df = self.term_structure.discount_factor(t)
        r = self.term_structure.risk_free_curve.instantaneous(0)
        s = self.s0
        sigma = self.sigma
        d1 = (1 / (sigma * (t**0.5))) * (np.log(s / k) + (r + 0.5 * sigma**2) * t)
        d2 = d1 - sigma * (t**0.5)
        return self.s0 * ndtr(d1) - k * df * ndtr(d2)

    def put(self, time_to_maturity: float, k: float) -> float:
        """Calculate the price of a put option using call-put parity."""
        t = time_to_maturity
        df = self.term_structure.discount_factor(t)
        return self.call(self.s0, t, k) - self.s0 + k * df


class BlackScholesModel(ItoProcess):
    """Black-Scholes model pricing by Monte-Carlo."""

    def __init__(self, term_structure: FlatTermStructure, s0: float, sigma: float):
        self.s0 = s0
        self.sigma = sigma
        self.term_structure = term_structure

    @ft.cached_property
    def mu_jit(self) -> callable:
        r = self.term_structure.risk_free_curve.instantaneous(0)

        @numba.njit
        def f(_: float, _xt: float) -> float:
            return bs_mu_jit(r, _, _xt)

        f(0, 0)  # compiles jit function
        return f

    @ft.cached_property
    def sigma_jit(self) -> callable:
        sig = self.sigma

        @numba.njit
        def f(_: float, _xt: float) -> float:
            return sig * _xt

        f(0, 0)  # compiles jit function
        return f

    @time_it
    def mc_exact(self, maturity, size, n_dyadic: int = N_DYADIC):
        σ = self.sigma
        r = self.term_structure.risk_free_curve.instantaneous(0)
        x0 = self.s0
        return bs_exact_mc(x0, r, σ, maturity, size, n_dyadic)


def bs_exact_mc(
    s0: float, r: float, sigma: float, t: float, size: int | tuple[int], n_dyadic: int
) -> Path:
    brownian = brownian_trajectories(t, size=size, n=n_dyadic)
    tk = brownian.t
    λ = r - 0.5 * sigma**2
    xk = s0 * np.exp(λ * tk + sigma * brownian.x)
    return Path(tk, xk)


def npv(
    _payoff: callable,
    model: ItoProcess,
    time_to_maturity: float,
    n_samples: int = 2**N_DYADIC,
):
    paths = model.mc_euler(time_to_maturity, size=n_samples)
    ts = model.term_structure
    return np.mean(ts.discount_factor(time_to_maturity) * _payoff(paths))


def main() -> None:
    """Compute."""

    K = 1.5
    R = 0.1
    T = 365
    s0 = 1
    sigma = 0.1
    n_mc = 2000
    risk_free = FlatForward(R, scale=DayCountConvention.One)
    ts = FlatTermStructure(risk_free)
    bs_model = BlackScholesModel(ts, s0, sigma)
    euler_paths = bs_model.mc_euler(size=n_mc, maturity=T)
    exact_paths = bs_model.mc_exact(size=n_mc, maturity=T)
    logger.info(euler_paths)
    logger.info(exact_paths)
    call_price_mc = npv(payoff.european_call(K), bs_model, time_to_maturity=T)
    # call_price_mc_exact = npv(payoff.european_call(bs, 0, T, K), exact_paths)
    call_price_det = BlackScholesModelDeterministic(ts, s0, sigma).call(T, K)

    logger.info(f"{call_price_mc=}")
    # logger.info(f"{call_price_mc_exact=}")
    logger.info(f"{call_price_det=}")
    # logger.info("------------------")

    # put_price_mc = npv(payoff.european_put(bs, 0, T, K), euler_paths)
    # put_price_mc_exact = npv(payoff.european_put(bs, 0, T, K), exact_paths)
    # put_price_det = BlackScholesModelDeterministic(bs).put(s0, T, K)

    # logger.info(f"{put_price_mc=}")
    # logger.info(f"{put_price_mc_exact=}")
    # logger.info(f"{put_price_det=}")


if __name__ == "__main__":
    main()
