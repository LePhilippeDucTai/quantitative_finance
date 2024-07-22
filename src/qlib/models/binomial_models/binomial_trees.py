import numpy as np
from numba import njit


@njit
def binomial_tree(x, N, u, d):
    """General binomial tree in an array structure
    x : starting point of the tree
    N : number of periods
    u : up multiplicative factor
    d : down multiplicative factor
    """
    p = np.arange(N)
    us, ds = u**p, d**p
    M = np.zeros((N, N))
    M[0, 0] = 1.0
    for j in range(N):
        for i in range(j + 1):
            M[i, j] = us[j - i] * ds[i]
    return M * x


def crr_ud(sigma, dt):
    """Cox-Ross-Rubinstein up and down factors given sigma and dt."""
    return np.exp(sigma * np.sqrt(dt)), np.exp(-sigma * np.sqrt(dt))


def jr_ud(r, sigma, dt):
    """Jarrow-Rudd up and down factors given r, sigma and dt."""
    m = np.exp((r - sigma**2 / 2) * dt)
    s = np.exp(sigma * np.sqrt(dt))
    u = m * s
    d = m / s
    return u, d


def crr_q(r, u, d, dt):
    """Risk neutral probability in the Cox-Ross-Rubinstein model,
    given r (short interest rate), u, d and dt.
    """
    R = np.exp(r * dt)
    return (R - d) / (u - d)


def jr_q():
    """Risk neutral probability in the Jarrow-Rudd model,
    given r (short interest rate), u, d and dt.
    """
    return 0.5


class BinomialTree:
    """General binomial Tree model. Not to be used explicitely."""

    def __init__(self, s0, r, sigma, T, dt):
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.u, self.d = 2, 0.5

    def lattice(self):
        return binomial_tree(self.s0, self.n_periods, self.u, self.d)

    @property
    def q(self):
        return 0.5

    @property
    def n_periods(self):
        return int(self.T // self.dt) + 1


class CRRModel(BinomialTree):
    """Cox-Ross-Rubinstein Model class."""

    def __init__(self, s0, r, sigma, T, dt):
        super().__init__(s0, r, sigma, T, dt)
        self.u, self.d = crr_ud(sigma, dt)

    @property
    def q(self):
        return crr_q(self.r, self.u, self.d, self.dt)


class JRModel(BinomialTree):
    """Jarrow-Rudd Model class"""

    def __init__(self, s0, r, sigma, T, dt):
        super().__init__(s0, r, sigma, T, dt)
        self.u, self.d = jr_ud(r, sigma, dt)

    @property
    def q(self):
        return jr_q()
