import numpy as np
from numba import njit


class ShortRateLatticeCustom:
    def __init__(self, lattice, q=0.5):
        self._lattice = np.array(lattice)
        self.q = q
        self.dt = 1.0

    def lattice(self):
        return self._lattice


@njit
def arrow_debreu_lattice(term_structure_lattice: np.ndarray, q: float):
    sr = term_structure_lattice
    lattice = np.zeros_like(sr)
    lattice[0, 0] = 1
    n = len(sr)
    for k in range(0, n - 1):
        lattice[k + 1, k + 1] = q * lattice[k, k] / (1 + sr[k, k])
        lattice[0, k + 1] = (1 - q) * lattice[0, k] / (1 + sr[0, k])
        for s in range(1, k + 1):
            left = lattice[s - 1, k] * q / (1 + sr[s - 1, k])
            right = lattice[s, k] * (1 - q) / (1 + sr[s, k])
            lattice[s, k + 1] = left + right
    return lattice
