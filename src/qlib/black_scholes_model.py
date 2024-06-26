"""Black-Scholes model for pricing options."""

import math
from dataclasses import dataclass

import numba
import numpy as np
from loguru import logger
from scipy.special import ndtr

from qlib.main import Path
from qlib.utils import to_tuple


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

    def mc_euler(
        self,
        x0: float,
        maturity: float,
        size: tuple[int],
        n_dyadic: int = 6,
    ) -> Path:
        size = to_tuple(size)
        n_t = maturity * 2**n_dyadic
        dt = 1 / (n_t - 1)
        g = np.random.default_rng().normal(scale=np.sqrt(dt), size=(*size, n_t))
        t = np.linspace(0, maturity, n_t)
        xt = np.zeros_like(g)
        xt[..., 0] = x0
        mu_jit = self.mu_jit()
        sigma_jit = self.sigma_jit()

        @numba.njit
        def inner(t: np.ndarray, xt: np.ndarray, n_t: int, g: np.ndarray) -> np.ndarray:
            for i in range(n_t - 1):
                mu = mu_jit(t[i], xt[..., i])
                sigma = sigma_jit(t[i], xt[..., i])
                xt[..., i + 1] = xt[..., i] + mu * dt + sigma * g[..., i]
            return xt

        xt = inner(t, xt, n_t, g)
        return Path(t, xt)


@dataclass
class BlackScholesParameters:
    """Parameters for the Black-Scholes model."""

    r: float
    sig: float


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
        d1 = (1 / (sigma * (t**0.5))) * (math.log(s / k) + (r + 0.5 * sigma**2) * t)
        d2 = d1 - sigma * (t**0.5)
        return s * ndtr(d1) - k * math.exp(-r * t) * ndtr(d2)

    def put(self, s: float, time_to_maturity: float, k: float) -> float:
        """Calculate the price of a put option using call-put parity."""
        t = time_to_maturity
        return self.call(t, k) - s + k * math.exp(-self.bs_params.r * t)


class BlackScholesModelMC(ItoProcess):
    """Black-Scholes model pricing by Monte-Carlo."""

    def __init__(self, bs_params: BlackScholesParameters):
        self.bs_params = bs_params

    def mu_jit(self) -> callable:
        r = self.bs_params.r

        @numba.njit
        def f(_: float, _xt: float) -> float:
            return r * _xt

        return f

    def sigma_jit(self) -> callable:
        sig = self.bs_params.sig

        @numba.njit
        def f(_: float, _xt: float) -> float:
            return sig * _xt

        return f


def main() -> None:
    """Compute."""
    bs = BlackScholesParameters(r=0.05, sig=0.1)
    bs_model_mc = BlackScholesModelMC(bs)
    path = bs_model_mc.mc_euler(1, size=10, maturity=1)
    logger.info(path)


if __name__ == "__main__":
    main()
