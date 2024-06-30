"""Black-Scholes model for pricing options."""

from dataclasses import dataclass

import numpy as np
from qlib.constant_parameters import DEFAULT_RNG, N_DYADIC, N_MC
from qlib.financial.payoffs import (
    AsianCallOption,
    EuropeanCallOption,
    EuropeanOptionParameters,
    EuropeanPutOption,
)
from qlib.models.brownian import Path, brownian_trajectories
from qlib.numerical.euler_scheme import ComputationKind
from qlib.traits import FlatForward, ItoProcessParameters, Model, TermStructure
from qlib.utils.logger import logger
from qlib.utils.timing import time_it
from scipy.special import ndtr


def bs_exact_mc(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    size: int | tuple[int],
    n_dyadic: int,
) -> Path:
    """Exact simulation of the underlying in the BS model."""
    brownian = brownian_trajectories(t, size=size, n=n_dyadic)
    tk = brownian.t
    λ = r - 0.5 * sigma**2
    xk = s0 * np.exp(λ * tk + sigma * brownian.x)
    return Path(tk, xk)


@dataclass
class BlackScholesParameters(ItoProcessParameters):
    x0: float
    sigma: float


class BlackScholesModel(Model):
    """Black-Scholes model pricing by Monte-Carlo."""

    def __init__(
        self, model_parameters: BlackScholesParameters, term_structure: TermStructure
    ):
        super().__init__(model_parameters, term_structure)
        self.model_parameters = model_parameters

    def mu(self, t, x: float):
        return x * self.r

    def sigma(self, _, x: float):
        return x * self.sig

    @property
    def r(self):
        return self.term_structure.rates_model.instantaneous()

    @property
    def x0(self):
        return self.model_parameters.x0

    @property
    def sig(self):
        return self.model_parameters.sigma

    @time_it
    def mc_exact(
        self,
        time_to_maturity: float,
        size: int = N_MC,
        n_dyadic: int = N_DYADIC,
    ):
        return bs_exact_mc(self.x0, self.r, self.sig, time_to_maturity, size, n_dyadic)

    def mc_terminal(self, time_to_maturity: float, size: int = N_MC):
        maturity = time_to_maturity
        r, σ = self.r, self.sig
        mu = (r - σ**2 / 2) * maturity
        s = self.sig * np.sqrt(maturity)
        exponential = DEFAULT_RNG.lognormal(mean=mu, sigma=s, size=(size, 1))
        time = np.array([maturity])
        return Path(time, self.x0 * exponential)

    def call(self, time_horizon: float, k: float) -> float:
        """Calculate the price of a call option."""
        s, t = self.x0, time_horizon
        r = self.r
        sigma = self.sig
        d1 = (1 / (sigma * (t**0.5))) * (np.log(s / k) + (r + 0.5 * sigma**2) * t)
        d2 = d1 - sigma * (t**0.5)
        return s * ndtr(d1) - k * np.exp(-r * t) * ndtr(d2)

    def put(self, time_horizon: int, k: float) -> float:
        """Calculate the price of a put option using call-put parity."""
        s, t = self.x0, time_horizon
        return self.call(k) - s + k * np.exp(-self.r * t)


def main():
    r = 0.05
    sig = 0.25
    x0 = 100
    maturity = 30 / 365
    strike_k = x0
    flat_curve = FlatForward(r)
    term_structure = TermStructure(flat_curve)
    bs_params = BlackScholesParameters(x0, sig)
    bs = BlackScholesModel(bs_params, term_structure)
    european_option_parameters = EuropeanOptionParameters(maturity, strike_k)

    option = EuropeanCallOption(bs, european_option_parameters)
    call_price_euler = option.npv(ComputationKind.EULER)
    call_price_exact = option.npv(ComputationKind.EXACT)
    call_price_terminal = option.npv(ComputationKind.TERMINAL)
    logger.info(f"{call_price_euler=}")
    logger.info(f"{call_price_exact=}")
    logger.info(f"{call_price_terminal=}")

    option = EuropeanPutOption(bs, european_option_parameters)
    put_price_euler = option.npv(ComputationKind.EULER)
    put_price_exact = option.npv(ComputationKind.EXACT)
    put_price_terminal = option.npv(ComputationKind.TERMINAL)
    logger.info(f"{put_price_euler=}")
    logger.info(f"{put_price_exact=}")
    logger.info(f"{put_price_terminal=}")

    option = AsianCallOption(bs, european_option_parameters)
    call_price_euler = option.npv(ComputationKind.EULER)
    call_price_exact = option.npv(ComputationKind.EXACT)
    logger.info(f"{call_price_euler=}")
    logger.info(f"{call_price_exact=}")


if __name__ == "__main__":
    main()
