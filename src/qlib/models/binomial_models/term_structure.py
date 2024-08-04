import functools as ft

import numpy as np
import scipy.optimize as so
from numba import njit
from qlib.models.binomial_models.binomial_trees import BinomialTree


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


@ft.cache
def _time_matrix(n_dates: int) -> np.ndarray:
    return (
        np.tile(np.arange(n_dates)[np.newaxis, :], (n_dates, 1))
        - np.arange(n_dates)[:, np.newaxis]
    )


def black_derman_toy_lattice(
    parameters: np.ndarray,
    n_dates: int,
) -> np.ndarray:
    """Parameters :
    parameters[0] -> array of a_i's
    parameters[1] -> array of b_i's
    """
    a, b = parameters
    a = a[:, np.newaxis] / 100
    b = b[:, np.newaxis]
    j = _time_matrix(n_dates)
    return np.triu(a * np.exp(b * j))


def zcb_prices(arrow_debreu_lattice: np.ndarray) -> np.ndarray:
    return np.sum(arrow_debreu_lattice[:, 1:], axis=0)


def spot_rates_from_zcb_prices(zcb_prices: np.ndarray) -> np.ndarray:
    """Solves ZCB_price(t) =  1 / (1 + rate) ** t
    gives : rate = 1 / ZCB ** (1 / t) - 1
    """
    dates = np.arange(1, len(zcb_prices) + 1)
    return zcb_prices ** (-1 / dates) - 1


def loss_function(marked: np.ndarray, model: np.ndarray) -> float:
    return np.linalg.norm(marked - model)


def black_derman_toy_loss(
    parameters: np.ndarray, marked_spot_rates: np.ndarray
) -> float:
    n_dates = len(marked_spot_rates) + 1
    params = parameters.reshape((2, -1))
    model_spot_rates = compute_black_derman_toy_spot_rates(n_dates, params)
    return loss_function(marked_spot_rates, model_spot_rates)


def compute_black_derman_toy_spot_rates(n_dates: int, params: np.ndarray):
    bdt = black_derman_toy_lattice(params, n_dates)
    elementary_prices = arrow_debreu_lattice(bdt, 0.5)
    zcb_p = zcb_prices(elementary_prices)
    return spot_rates_from_zcb_prices(zcb_p)


def black_derman_toy_calibration(marked_spot_rates: np.ndarray) -> np.ndarray:
    n_dates = len(marked_spot_rates) + 1
    a = 5 * np.ones(n_dates)
    b = 0.005 * np.ones(n_dates)
    x0 = np.array([a, b]).flatten()
    return so.minimize(black_derman_toy_loss, x0=x0, args=(marked_spot_rates,))


def main():
    r0 = 0.06
    u, d = 1.25, 0.9
    t_max = 5
    term_structure = BinomialTree(r0, t_max, u=u, d=d)
    print(term_structure.lattice())

    # Marked spot rates observed from year 1 to year 14
    marked_spot_rates = (
        np.array(
            [
                7.3,
                7.62,
                8.1,
                8.45,
                9.2,
                9.64,
                10.12,
                10.45,
                10.75,
                11.22,
                11.55,
                11.92,
                12.2,
                12.32,
            ]
        )
        / 100
    )
    print(marked_spot_rates)
    ad = arrow_debreu_lattice(term_structure.lattice(), 0.5)

    n_dates = len(marked_spot_rates) + 1
    a = 5 * np.ones(n_dates) / 100
    b = 0.005 * np.ones(n_dates)
    parameters = np.array([a, b])
    loss = black_derman_toy_loss(parameters, marked_spot_rates)
    # print(loss)
    calib = black_derman_toy_calibration(marked_spot_rates)
    x_opt = calib.x.reshape((2, -1))
    print(calib)
    a, b = x_opt
    optimal_spot_rates = compute_black_derman_toy_spot_rates(n_dates, x_opt)
    bdt = black_derman_toy_lattice(x_opt, n_dates)
    print(optimal_spot_rates)
    print(marked_spot_rates)
    print(bdt)
    # print(x_opt)


if __name__ == "__main__":
    main()
