"""Black-Scholes model for pricing options."""

from dataclasses import dataclass

import numpy as np
import scipy.stats as ss
from qlib.constant_parameters import DEFAULT_RNG, N_DYADIC, N_MC
from qlib.financial.payoffs import PricingData
from qlib.models.brownian import Path, brownian_trajectories
from qlib.traits import ItoProcessParameters, Model, TermStructure
from qlib.utils.timing import time_it
from scipy.special import ndtr


def bs_exact_mc(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    size: int | tuple[int],
    n_dyadic: int,
    generator: np.random.Generator,
) -> Path:
    """Exact simulation of the underlying in the BS model."""
    brownian = brownian_trajectories(t, size=size, n=n_dyadic, gen=generator)
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
        generator: np.random.Generator = DEFAULT_RNG,
    ):
        return bs_exact_mc(
            self.x0, self.r, self.sig, time_to_maturity, size, n_dyadic, generator
        )

    def mc_terminal(
        self,
        time_to_maturity: float,
        size: int = N_MC,
        generator: np.random.Generator = DEFAULT_RNG,
    ):
        maturity = time_to_maturity
        r, σ = self.r, self.sig
        mu = (r - σ**2 / 2) * maturity
        s = self.sig * np.sqrt(maturity)
        exponential = generator.lognormal(mean=mu, sigma=s, size=(size, 1))
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

    def delta(self, time_horizon: float, k: float) -> float:
        _d1 = d1(self.x0, k, self.r, self.sig, time_horizon)
        return ndtr(_d1)

    def gamma(self, time_horizon: float, k: float) -> float:
        _d1 = d1(self.x0, k, self.r, self.sig, time_horizon)
        return ss.norm.pdf(_d1) / (self.sig * self.x0 * np.sqrt(time_horizon))

    def vega(self, time_horizon, k: float) -> float:
        _d1 = d1(self.x0, k, self.r, self.sig, time_horizon)
        return self.x0 * np.sqrt(time_horizon) * ss.norm.pdf(_d1)

    def call_det_pricing(self, time_horizon: float, k: float) -> PricingData:
        price = self.call(time_horizon, k)
        std = 0.0
        delta = self.delta(time_horizon, k)
        gamma = self.gamma(time_horizon, k)
        vega = self.vega(time_horizon, k)
        return PricingData(price, std, delta, gamma, vega)


def d1(s: float, k: float, r: float, sigma: float, t: float) -> float:
    return (1 / (sigma * (t**0.5))) * (np.log(s / k) + (r + 0.5 * sigma**2) * t)


def d2(s: float, k: float, r: float, sigma: float, t: float) -> float:
    _d1 = d1(s, k, r, sigma, t)
    return _d1 - sigma * (t**0.5)


def _test():
    pass
    # option = EuropeanPutOption(bs, european_option_parameters)
    # put_price_euler = option.npv(ComputationKind.EULER)
    # put_price_exact = option.npv(ComputationKind.EXACT)
    # put_price_terminal = option.npv(ComputationKind.TERMINAL)
    # logger.info(f"{put_price_euler=}")
    # logger.info(f"{put_price_exact=}")
    # logger.info(f"{put_price_terminal=}")

    # option = AsianCallOption(bs, european_option_parameters)
    # call_price_euler = option.npv(ComputationKind.EULER)
    # call_price_exact = option.npv(ComputationKind.EXACT)
    # logger.info(f"{call_price_euler=}")
    # logger.info(f"{call_price_exact=}")


# if __name__ == "__main__":
#     main()
