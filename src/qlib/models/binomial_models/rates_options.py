import numpy as np
from numba import njit
from qlib.models.binomial_models.binomial_trees import BinomialTree
from qlib.models.binomial_models.european_options import (
    EuropeanCallOption,
    EuropeanOption,
    rn_expectation,
)
from qlib.models.binomial_models.term_structure import arrow_debreu_lattice


class FuturesOption(EuropeanOption):
    def __init__(self, maturity: int, model: BinomialTree):
        super().__init__(maturity, model, None)
        self.term_structure = BinomialTree(0.0, maturity, dt=1, u=0.5, d=0.5)


class CapletOption(EuropeanCallOption):
    def __init__(
        self,
        maturity: int,
        model: BinomialTree,
        term_structure: BinomialTree,
        K_strike: float,
        in_arears: bool = True,
    ):
        super().__init__(maturity, model, term_structure, K_strike)
        self.in_arears = in_arears

    def payoff(self, x):
        if self.in_arears:
            discount = 1 / (1 + x)
        else:
            discount = 1
        return super().payoff(x) * discount


@njit
def compute_agg_induction(
    N, V: np.ndarray, sr_lattice, q: float, dt: float
) -> np.ndarray:
    payoffs = V.copy()
    for j in range(N - 1, 0, -1):
        for i in range(j):
            r = sr_lattice[i, j - 1]
            expected_value = rn_expectation(V[i, j], V[i + 1, j], r, q, dt)
            V[i, j - 1] = payoffs[i, j - 1] + expected_value
    return V


class SwapDerivative(EuropeanOption):
    def __init__(
        self, maturity: int, term_structure: BinomialTree, swap_rate: float, payer: bool
    ):
        """Swap derivative class.
        Payer means paying fixed rate, receiving floating rate. (r - K)
        Payer == False means receive fixed rate, paying floating rate (K - r).
        """
        super().__init__(maturity, term_structure, term_structure)
        self.swap_rate = swap_rate
        self.payer = payer
        self.orient = 1 if payer else -1

    def payoff(self, x: np.ndarray) -> np.ndarray:
        return self.orient * (x - self.swap_rate) / (1 + x)

    def compute_induction(
        self, N: int, V: np.ndarray, short_rate_lattice: np.ndarray, q: float, dt: float
    ) -> np.ndarray:
        return compute_agg_induction(N, V, short_rate_lattice, q, dt)


class Swaption(EuropeanCallOption):
    def __init__(
        self, swap_maturity: int, swaption_maturity, swap_rate: float, term_structure
    ):
        """Swaption.
        Swap_maturity :
        """
        K_strike = 0.0
        self.swaption_maturity = swaption_maturity
        swap_derivative = SwapDerivative(
            maturity=swap_maturity,
            term_structure=term_structure,
            swap_rate=swap_rate,
            payer=True,
        )
        super().__init__(
            maturity=swaption_maturity,
            model=swap_derivative,
            term_structure=term_structure,
            K_strike=K_strike,
        )


class ForwardSwap:
    def __init__(
        self,
        term_structure: BinomialTree,
        swap_rate: float,
        notional: float,
        start: int,
        end: int,
        payer: bool = True,
    ):
        self.term_structure = term_structure
        self.swap_rate = swap_rate
        self.orient = 1 if payer else -1
        self.start = start
        self.end = end
        self.notional = notional

    def payoff(self, r):
        return self.orient * (r - self.swap_rate) / (1 + r)

    def payoff_values(self):
        short_rates_lattice = self.term_structure.lattice()
        return np.triu(self.payoff(short_rates_lattice))

    def npv(self):
        a, b = self.start, self.end
        ts = self.term_structure.lattice()
        q = self.term_structure.q
        ad = arrow_debreu_lattice(ts, q)[:, a:b]
        values = self.payoff_values()[:, a:b]
        print(ad.round(4))
        print(values.round(4))
        return self.notional * (ad * values).sum()
