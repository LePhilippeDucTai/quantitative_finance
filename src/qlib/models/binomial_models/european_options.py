import numpy as np
from numba import njit
from qlib.models.binomial_models.binomial_trees import BinomialTree


@njit
def rn_expectation(u, d, r, q, dt):
    if dt == 1:
        df = 1 / (1 + r)
    else:
        df = np.exp(-r * dt)
    return (q * u + (1 - q) * d) * df


@njit
def compute_induction(N, V, sr_lattice, q, dt):
    for j in range(N - 1, 0, -1):
        for i in range(j):
            r = sr_lattice[i, j - 1]
            expected_value = rn_expectation(V[i, j], V[i + 1, j], r, q, dt)
            V[i, j - 1] = expected_value
    return V


class EuropeanOption:
    def __init__(
        self, maturity: int, model: BinomialTree, term_structure: BinomialTree
    ):
        self.model = model
        self.maturity = maturity
        self.term_structure = term_structure
        self._npv_lattice = None
        self.dt = model.dt
        self.q = model.q

    def payoff(self, x):
        return x

    def compute_induction(self, N, V, short_rate_lattice, q, dt):
        return compute_induction(N, V, short_rate_lattice, q, dt)

    def npv(self):
        model_lattice = self.model.lattice()
        short_rate_lattice = self.term_structure.lattice()
        N = int(self.maturity // self.dt) + 1
        V = np.triu(self.payoff(model_lattice[:N, :N]))
        dt, q = self.dt, self.q
        self._npv_lattice = self.compute_induction(N, V, short_rate_lattice, q, dt)
        return self._npv_lattice[0, 0]

    def lattice(self):
        self.npv()
        return self._npv_lattice


class EuropeanCallOption(EuropeanOption):
    def __init__(
        self,
        maturity: int,
        model: BinomialTree,
        term_structure: BinomialTree,
        K_strike: float,
    ):
        super().__init__(maturity, model, term_structure)
        self.K_strike = K_strike

    def payoff(self, x):
        return np.maximum(x - self.K_strike, 0)


class EuropeanPutOption(EuropeanOption):
    def __init__(
        self,
        maturity: int,
        model: BinomialTree,
        term_structure: BinomialTree,
        K_strike: float,
    ):
        super().__init__(maturity, model, term_structure)
        self.K_strike = K_strike

    def payoff(self, x):
        return np.maximum(self.K_strike - x, 0)
