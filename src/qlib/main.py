"""Main module."""

import numpy as np

from qlib.financial.payoffs import EuropeanCallOption, EuropeanOptionParameters
from qlib.models.black_scholes_model import BlackScholesModel, BlackScholesParameters
from qlib.models.brownian import brownian_trajectories, plot_brownians
from qlib.numerical.euler_scheme import ComputationKind
from qlib.traits import FlatForward, TermStructure
from qlib.utils.logger import logger

logger.level("INFO")


def main_0() -> None:
    """Test."""
    p = brownian_trajectories(1, size=100, n=10)
    plot_brownians(p)


def main():
    r = 0.05
    sig = 0.20
    x0 = 100
    maturity = 1
    strike_k = 100
    flat_curve = FlatForward(r)
    term_structure = TermStructure(flat_curve)
    bs_params = BlackScholesParameters(x0, sig)
    bs = BlackScholesModel(bs_params, term_structure)
    european_option_parameters = EuropeanOptionParameters(maturity, strike_k)

    option = EuropeanCallOption(bs, european_option_parameters)
    call_price_euler = option.npv(kind=ComputationKind.EULER)
    call_price_exact = option.npv(kind=ComputationKind.EXACT)
    call_price_terminal = option.npv(kind=ComputationKind.TERMINAL)

    call_price_milstein = option.npv(kind=ComputationKind.MILTSTEIN)
    logger.info(f"{call_price_euler=}")
    logger.info(f"{call_price_exact=}")
    logger.info(f"{call_price_terminal=}")
    logger.info(f"{call_price_milstein=}")

    call_pricing = option.pricing(seed_seq=np.random.SeedSequence(4599412))
    logger.info(f"{call_pricing=}")

    call_det_pricing = bs.call_det_pricing(maturity, strike_k)
    logger.info(f"{call_det_pricing=}")


if __name__ == "__main__":
    # main_0()
    main()
