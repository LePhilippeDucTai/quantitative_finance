"""Black-Scholes model for pricing options."""

import math
from dataclasses import dataclass

from scipy.special import ndtr


@dataclass
class BlackScholesModelParameters:
    """Parameters for the Black-Scholes model."""

    r: float
    sigma: float


class BlackScholesModel:
    """Black-Scholes model for pricing options."""

    def __init__(self, bs_params: BlackScholesModelParameters) -> None:
        """Initialize the model with a BlackScholesModelParameters object."""
        self.bs_params = bs_params

    def call(self, s: float, time_to_maturity: float, k: float) -> float:
        """Calculate the price of a call option."""
        t = time_to_maturity
        r = self.bs_params.r
        sigma = self.bs_params.sigma
        d1 = (1 / (sigma * (t**0.5))) * (math.log(s / k) + (r + 0.5 * sigma**2) * t)
        d2 = d1 - sigma * (t**0.5)
        return s * ndtr(d1) - k * math.exp(-r * t) * ndtr(d2)

    def put(self, s: float, time_to_maturity: float, k: float) -> float:
        """Calculate the price of a put option using call-put parity."""
        t = time_to_maturity
        return self.call(t, k) - s + k * math.exp(-self.bs_params.r * t)
