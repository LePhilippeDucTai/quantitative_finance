"""Black-Scholes model for pricing options."""

from enum import Enum, auto

import numpy as np
from qlib.constant_parameters import DEFAULT_RNG, N_DYADIC, N_MC
from qlib.financial import payoffs
from qlib.models.brownian import Path, brownian_trajectories
from qlib.traits import Model
from qlib.utils.logger import logger
from qlib.utils.timing import time_it
from scipy.special import ndtr


class ComputationKind(Enum):
    DET = auto()
    EULER = auto()
    EULER_JIT = auto()
    EXACT = auto()
    MILTSTEIN = auto()
    TERMINAL = auto()


def bs_mu(t, x, r, sigma, *args):
    return r * x


def bs_sigma(t, x, r, sigma, *args):
    return sigma * x


class BlackScholesModel(Model):
    """Black-Scholes model pricing by Monte-Carlo."""

    def __init__(
        self, x0: float, time_horizon: float, risk_free_rate: float, sigma: float
    ):
        """Risk_free_rate: will become TermStructure in the future."""
        super().__init__(x0, time_horizon)
        self.r = risk_free_rate
        self.sig = sigma
        self.model_args = (risk_free_rate, sigma)

    def mu(self):
        return bs_mu

    def sigma(self):
        return bs_sigma

    @time_it
    def mc_exact(
        self,
        size: int = N_MC,
        n_dyadic: int = N_DYADIC,
    ):
        x0 = self.x0
        r = self.r
        σ = self.sig
        return bs_exact_mc(x0, r, σ, self.time_horizon, size, n_dyadic)

    def mc_terminal(self, size: int = N_MC):
        maturity = self.time_horizon
        mu = (self.r - self.sig**2 / 2) * maturity
        s = self.sig * np.sqrt(maturity)
        exponential = DEFAULT_RNG.lognormal(mean=mu, sigma=s, size=(size, 1))
        time = np.array([maturity])
        return Path(time, self.x0 * exponential)

    def call(self, k: float) -> float:
        """Calculate the price of a call option."""
        s, t = self.x0, self.time_horizon
        r = self.r
        sigma = self.sig
        d1 = (1 / (sigma * (t**0.5))) * (np.log(s / k) + (r + 0.5 * sigma**2) * t)
        d2 = d1 - sigma * (t**0.5)
        return s * ndtr(d1) - k * np.exp(-r * t) * ndtr(d2)

    def put(self, k: float) -> float:
        """Calculate the price of a put option using call-put parity."""
        s, t = self.x0, self.time_horizon
        return self.call(k) - s + k * np.exp(-self.r * t)


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


def method_selector(model: BlackScholesModel, method: ComputationKind) -> callable:
    mapping = {
        ComputationKind.EULER: model.mc_euler,
        ComputationKind.EULER_JIT: model.mc_euler_jit,
        ComputationKind.EXACT: model.mc_exact,
        ComputationKind.TERMINAL: model.mc_terminal,
    }
    return mapping[method]


class EuropeanOption:
    def __init__(self, model: BlackScholesModel, strike_k: float, maturity: float):
        self.strike_k = strike_k
        self.maturity = maturity
        self.model = model

    def call_npv(self, method: ComputationKind, n_mc: int = N_MC):
        if method == ComputationKind.DET:
            return self.model.call(self.strike_k)
        simulation_function = method_selector(self.model, method)
        sample_paths: Path = simulation_function(size=n_mc)
        r = self.model.r
        h = payoffs.call(sample_paths.x[..., -1], r, self.strike_k, self.maturity)
        return np.mean(h)

    def put_npv(self, method: ComputationKind, n_mc: int = N_MC):
        c0 = self.call_npv(method, n_mc)
        r, t = self.model.r, self.maturity
        return c0 - self.model.x0 + self.strike_k * np.exp(-r * t)


class BarrierOption:
    def __init__(self, strike_k: float, maturity: float, barrier: float):
        self.strike_k = strike_k
        self.maturity = maturity
        self.barrier = barrier


class AsianOption:
    def __init__(self, strike_k: float, maturity: float):
        self.strik_k = strike_k
        self.maturity = maturity


def main() -> None:
    """Compute and test."""
    s0 = 100
    strike_k = s0
    rfr = 0.1
    tmt = 2
    # barrier = 20
    sigma = 0.5

    bs_model_mc = BlackScholesModel(s0, tmt, rfr, sigma)
    european = EuropeanOption(bs_model_mc, strike_k, tmt)
    call_price_det = european.call_npv(ComputationKind.DET)
    call_price_euler = european.call_npv(ComputationKind.EULER)
    call_price_euler_jit = european.call_npv(ComputationKind.EULER_JIT)
    call_price_terminal = european.call_npv(ComputationKind.TERMINAL)

    logger.info(f"{call_price_det=}")
    logger.info(f"{call_price_terminal=}")
    logger.info(f"{call_price_euler=}")
    logger.info(f"{call_price_euler_jit=}")
    # call_price_mc = npv(payoff.call, euler_paths, r=rfr, k_strike=strike_k, tmt=tmt)
    # call_price_mc_exact = npv(
    #     payoff.call, exact_paths, r=rfr, k_strike=strike_k, tmt=tmt
    # )
    # call_price_det = bs_model_det.call(s0, tmt, strike_k)

    # logger.info(f"{call_price_mc=}")
    # logger.info(f"{call_price_mc_exact=}")
    # logger.info(f"{call_price_det=}")
    # logger.info("------------------")

    # put_price_mc = npv(payoff.put, euler_paths, r=rfr, k_strike=strike_k, tmt=tmt)
    # put_price_mc_exact = npv(payoff.put, exact_paths, r=rfr, k_strike=strike_k, tmt=tmt)  # noqa: E501
    # put_price_det = bs_model_det.put(s0, tmt, strike_k)

    # logger.info(f"{put_price_mc=}")
    # logger.info(f"{put_price_mc_exact=}")
    # logger.info(f"{put_price_det=}")

    # logger.info("-----------------")
    # asian_call_price_mc = npv(
    #     payoff.asian_call,
    #     euler_paths,
    #     r=rfr,
    #     k_strike=strike_k,
    #     tmt=tmt,
    # )
    # logger.info(f"{asian_call_price_mc=}")
    # logger.info("-----------------")
    # barrier_uo_call_price_mc = npv(
    #     payoff.up_n_out_call,
    #     euler_paths,
    #     r=rfr,
    #     k_strike=strike_k,
    #     barrier=barrier,
    #     tmt=tmt,
    # )
    # logger.info(f"{barrier_uo_call_price_mc=}")

    # logger.info("-----------------")
    # barrier_do_call_price_mc = npv(
    #     payoff.down_n_out_call,
    #     euler_paths,
    #     r=rfr,
    #     k_strike=strike_k,
    #     barrier=5,
    #     tmt=tmt,
    # )
    # logger.info(f"{barrier_do_call_price_mc=}")

    # logger.info("-----------------")
    # digital_option_price = npv(
    #     payoff.digital_option,
    #     euler_paths,
    #     r=rfr,
    #     k_strike=strike_k,
    #     tmt=tmt,
    # )
    # logger.info(f"{digital_option_price=}")


if __name__ == "__main__":
    main()
