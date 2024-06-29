"""Black-Scholes model for pricing options."""

import numpy as np
from qlib import payoff
from qlib.constant_parameters import DEFAULT_RNG, N_DYADIC, N_MC
from qlib.models.brownian import Path, brownian_trajectories
from qlib.traits import EulerSchema, EulerSchemaJit
from qlib.utils.logger import logger
from qlib.utils.timing import time_it
from scipy.special import ndtr


def bs_mu(t, x, r, sigma, *args):
    return r * x


def bs_sigma(t, x, r, sigma, *args):
    return sigma * x


class BlackScholesModel(EulerSchemaJit, EulerSchema):
    """Black-Scholes model pricing by Monte-Carlo."""

    def __init__(self, risk_free_rate: float, sigma: float):
        """Risk_free_rate: will become TermStructure in the future."""
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
        x0: float,
        maturity: float,
        size: float,
        n_dyadic: int = N_DYADIC,
    ):
        r = self.r
        σ = self.sig
        return bs_exact_mc(x0, r, σ, maturity, size, n_dyadic)

    def mc_terminal(self, x0: float, maturity: float, size: float):
        mu = (self.r - self.sig**2 / 2) * maturity
        s = self.sig * np.sqrt(maturity)
        exponential = DEFAULT_RNG.lognormal(mean=mu, sigma=s, size=(size, 1))
        time = np.array([maturity])
        return Path(time, x0 * exponential)


class BlackScholesModelDeterministic(BlackScholesModel):
    """Black-Scholes model for pricing options."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the model with a BlackScholesModelParameters object."""
        super().__init__(*args, **kwargs)

    def call(self, s: float, time_to_maturity: float, k: float) -> float:
        """Calculate the price of a call option."""
        t = time_to_maturity
        r = self.r
        sigma = self.sig
        d1 = (1 / (sigma * (t**0.5))) * (np.log(s / k) + (r + 0.5 * sigma**2) * t)
        d2 = d1 - sigma * (t**0.5)
        return s * ndtr(d1) - k * np.exp(-r * t) * ndtr(d2)

    def put(self, s: float, time_to_maturity: float, k: float) -> float:
        """Calculate the price of a put option using call-put parity."""
        t = time_to_maturity
        return self.call(s, t, k) - s + k * np.exp(-self.r * t)


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


class EuropeanOption:
    def __init__(self, strike_k: float, maturity: float):
        self.strike_k = strike_k
        self.maturity = maturity

    def call_npv(self, sample_paths: Path, r: float):
        h = payoff.call(sample_paths.x[..., -1], r, self.strike_k, self.maturity)
        return np.mean(h)

    def put_npv(self, sample_paths: Path, r: float):
        h = payoff.put(sample_paths.x[..., -1], r, self.strike_k, self.maturity)
        return np.mean(h)


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
    n_mc = N_MC

    bs_model_mc = BlackScholesModel(rfr, sigma)
    bs_model_det = BlackScholesModelDeterministic(rfr, sigma)
    european = EuropeanOption(strike_k, tmt)
    euler_paths = bs_model_mc.mc_euler_jit(s0, size=n_mc, time_horizon=tmt)
    exact_paths = bs_model_mc.mc_exact(s0, size=n_mc, maturity=tmt)
    terminal_paths = bs_model_mc.mc_terminal(s0, size=n_mc, maturity=tmt)

    call_price_terminal = european.call_npv(terminal_paths, rfr)
    call_price_euler = european.call_npv(euler_paths, rfr)
    call_price_explicit = european.call_npv(exact_paths, rfr)
    call_price_det = bs_model_det.call(s0, tmt, strike_k)
    logger.info(f"{call_price_terminal=}")
    logger.info(f"{call_price_euler=}")
    logger.info(f"{call_price_det=}")
    logger.info(f"{call_price_explicit=}")
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
