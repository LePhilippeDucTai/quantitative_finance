"""Black-Scholes model for pricing options."""

import functools as ft
from dataclasses import dataclass

import numba
import numpy as np
from scipy.special import ndtr

from qlib.brownian import Path, TimeGrid, brownian_trajectories
from qlib.utils.misc import to_tuple
from qlib.utils.timing import time_it

N_DYADIC = 10  # Gives by default for each time interval [t, t + 1], 2 ** 6 = 64 points


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
class BlackScholesParameters:
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

    def mu_jit(self):
        return 0

    def sigma_jit(self):
        return 0

    @time_it
    def mc_euler(
        self,
        x0: float,
        maturity: float,
        size: tuple[int],
        n_dyadic: int = 6,
    ) -> Path:
        time_grid = TimeGrid(maturity, n_dyadic)
        dt, n_t, t = time_grid.dt, time_grid.n_dates, time_grid.t
        size = to_tuple(size)
        g = np.random.default_rng().normal(scale=np.sqrt(dt), size=(*size, n_t))
        xt = np.empty_like(g)

        mu_jit = self.mu_jit
        sigma_jit = self.sigma_jit
        xt[..., 0] = x0
        xt = euler_discretization(mu_jit, sigma_jit, t, xt, n_t, g, dt)
        return Path(t, xt)


class BlackScholesModelDeterministic:
    """Black-Scholes model for pricing options."""

    def __init__(self, bs: BlackScholesParameters) -> None:
        """Initialize the model with a BlackScholesModelParameters object."""
        self.bs_params = bs

    def call(self, s: float, time_to_maturity: float, k: float) -> float:
        """Calculate the price of a call option."""
        t = time_to_maturity
        r = self.bs_params.r
        sigma = self.bs_params.sig
        d1 = (1 / (sigma * (t**0.5))) * (np.log(s / k) + (r + 0.5 * sigma**2) * t)
        d2 = d1 - sigma * (t**0.5)
        return s * ndtr(d1) - k * np.exp(-r * t) * ndtr(d2)

    def put(self, s: float, time_to_maturity: float, k: float) -> float:
        """Calculate the price of a put option using call-put parity."""
        t = time_to_maturity
        return self.call(t, k) - s + k * np.exp(-self.bs_params.r * t)


class BlackScholesModelMC(ItoProcess):
    """Black-Scholes model pricing by Monte-Carlo."""

    def __init__(self, bs_params: BlackScholesParameters):
        self.bs_params = bs_params

    @ft.cached_property
    def mu_jit(self) -> callable:
        r = self.bs_params.r

        @numba.njit
        def f(_: float, _xt: float) -> float:
            return bs_mu_jit(r, _, _xt)

        f(0, 0)  # compiles jit function
        return f

    @ft.cached_property
    def sigma_jit(self) -> callable:
        sig = self.bs_params.sig

        @numba.njit
        def f(_: float, _xt: float) -> float:
            return sig * _xt

        f(0, 0)  # compiles jit function
        return f

    @time_it
    def mc_exact(self, x0, maturity, size, n_dyadic: int = 6):
        r = self.bs_params.r
        σ = self.bs_params.sig
        return bs_exact_mc(x0, r, σ, maturity, size, n_dyadic)


def bs_exact_mc(
    s0: float, r: float, sigma: float, t: float, size: int | tuple[int], n_dyadic: int
) -> Path:
    brownian = brownian_trajectories(t, size=size, n=n_dyadic)
    tk = brownian.t
    λ = r - 0.5 * sigma**2
    xk = s0 * np.exp(λ * tk + sigma * brownian.x)
    return Path(tk, xk)


def main() -> None:
    """Compute."""
    bs = BlackScholesParameters(r=0.1, sig=0.1)
    bs_model_mc = BlackScholesModelMC(bs)
    path = bs_model_mc.mc_euler(1, size=1000, maturity=5)
    s = bs_model_mc.mc_exact(3, size=1000, maturity=5)
    path.plot()
    s.plot()


if __name__ == "__main__":
    main()