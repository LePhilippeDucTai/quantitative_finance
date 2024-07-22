import numpy as np
from numba import njit
from qlib.models.binomial_models.binomial_trees import BinomialTree


@njit
def rn_expectation(u, d, r, q, dt):
    return (q * u + (1 - q) * d) * np.exp(-r * dt)


@njit
def compute_induction(N, V, r, q, dt):
    for j in range(N - 1, 0, -1):
        for i in range(j):
            expected_value = rn_expectation(V[i, j], V[i + 1, j], r, q, dt)
            V[i, j - 1] = expected_value
    return V


class EuropeanOption:
    def __init__(self, model: BinomialTree):
        self.model = model
        self.npv_lattice = None

    def payoff(self, x):
        return x

    def npv(self):
        lattice = self.model.lattice()
        terminal_value = lattice[:, -1]
        r, dt, q, N = self.model.r, self.model.dt, self.model.q, self.model.n_periods
        V = np.zeros((N, N))
        V[:, -1] = self.payoff(terminal_value)
        self.npv_lattice = compute_induction(N, V, r, q, dt)
        return self.npv_lattice[0, 0]


class EuropeanCallOption(EuropeanOption):
    def __init__(self, model, K_strike):
        self.model = model
        self.K_strike = K_strike

    def payoff(self, x):
        return np.maximum(x - self.K_strike, 0)


class EuropeanPutOption(EuropeanOption):
    def __init__(self, model, K_strike):
        self.model = model
        self.K_strike = K_strike

    def payoff(self, x):
        return np.maximum(self.K_strike - x, 0)
