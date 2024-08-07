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
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    return u, d


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

    def __init__(self, s0, T, dt: float = 1.0, u: float = 2.0, d: float = 0.5):
        self.s0 = s0
        self.T = T
        self.dt = dt
        self.u, self.d = u, d

    def lattice(self):
        return binomial_tree(self.s0, self.n_periods, self.u, self.d)

    @property
    def q(self):
        return 0.5

    @property
    def n_periods(self):
        return int(self.T // self.dt) + 1


class FlatForward(BinomialTree):
    def __init__(self, s0, T, dt: float = 1.0, u: float = 1, d: float = 1):
        super().__init__(s0, T, dt, u, d)


class CRRModel(BinomialTree):
    """Cox-Ross-Rubinstein Model class."""

    def __init__(self, s0, sigma, flat_ts: FlatForward, T, dt):
        super().__init__(s0, T, dt)
        self.sigma = sigma
        self.r = flat_ts.s0
        self.u, self.d = crr_ud(sigma, dt)

    @property
    def q(self):
        return crr_q(self.r, self.u, self.d, self.dt)


class JRModel(BinomialTree):
    """Jarrow-Rudd Model class"""

    def __init__(self, s0, sigma, flat_ts: FlatForward, T, dt):
        super().__init__(s0, T, dt)
        self.r = flat_ts.s0
        self.sigma = sigma
        self.u, self.d = jr_ud(self.r, sigma, dt)

    @property
    def q(self):
        return jr_q()
