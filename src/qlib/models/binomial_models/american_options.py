import numpy as np
from numba import njit
from qlib.models.binomial_models.binomial_trees import BinomialTree
from qlib.models.binomial_models.european_options import EuropeanOption, rn_expectation


@njit
def compute_max_induction(N, V: np.ndarray, sr_lattice, q, dt):
    payoffs = V.copy()
    for j in range(N - 1, 0, -1):
        for i in range(j):
            r = sr_lattice[i, j - 1]
            expected_value = rn_expectation(V[i, j], V[i + 1, j], r, q, dt)
            V[i, j - 1] = max(payoffs[i, j - 1], expected_value)
    return V


class AmericanOption(EuropeanOption):
    def __init__(
        self, maturity: int, model: BinomialTree, term_structure: BinomialTree
    ):
        super().__init__(maturity, model, term_structure)

    def compute_induction(self, N, V, short_rate_lattice, q, dt):
        return compute_max_induction(N, V, short_rate_lattice, q, dt)


class AmericanCallOption(AmericanOption):
    def __init__(self, maturity, model, term_structure, K_strike):
        super().__init__(maturity, model, term_structure)
        self.K_strike = K_strike

    def payoff(self, x):
        return np.maximum(x - self.K_strike, 0)


class AmericanPutOption(AmericanOption):
    def __init__(self, maturity, model, term_structure, K_strike):
        super().__init__(maturity, model, term_structure)
        self.K_strike = K_strike

    def payoff(self, x):
        return np.maximum(self.K_strike - x, 0)
