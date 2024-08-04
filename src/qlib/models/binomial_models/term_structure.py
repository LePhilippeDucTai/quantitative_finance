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
    n, _ = sr.shape
    lattice = np.zeros((n + 1, n + 1))
    lattice[0, 0] = 1
    for k in range(0, n):
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
    a = a[:, np.newaxis]
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
    return np.linalg.norm(marked - model) ** 2


def black_derman_toy_loss(
    x: np.ndarray,
    marked_spot_rates: np.ndarray,
    a_fixed: np.ndarray | None,
    b_fixed: np.ndarray | None,
) -> float:
    n_dates = len(marked_spot_rates)
    if (a_fixed is None) and (b_fixed is not None):
        x = np.array([x, b_fixed]).flatten()
    if (a_fixed is not None) and (b_fixed is None):
        x = np.array([a_fixed, x]).flatten()
    params = x.reshape((2, -1))
    model_spot_rates = compute_black_derman_toy_spot_rates(n_dates, params)
    return loss_function(marked_spot_rates * 100, model_spot_rates * 100)


def compute_black_derman_toy_spot_rates(n_dates: int, params: np.ndarray):
    bdt = black_derman_toy_lattice(params, n_dates)
    elementary_prices = arrow_debreu_lattice(bdt, 0.5)
    zcb_p = zcb_prices(elementary_prices)
    return spot_rates_from_zcb_prices(zcb_p)


def black_derman_toy_calibration(
    marked_spot_rates: np.ndarray,
    a_fixed: np.ndarray | None = None,
    b_fixed: np.ndarray | None = None,
) -> np.ndarray:
    n_dates = len(marked_spot_rates)
    a = np.random.default_rng().uniform(size=n_dates + 1)
    b = np.random.default_rng().uniform(size=n_dates + 1)
    if a_fixed is not None:
        x0 = b
    elif b_fixed is not None:
        x0 = a
    else:
        x0 = np.array([a, b]).flatten()
    n = len(x0)
    return so.minimize(
        black_derman_toy_loss,
        x0=x0,
        args=(marked_spot_rates, a_fixed, b_fixed),
        method="L-BFGS-B",
        bounds=[(0, None)] * n,
    )


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
    print(ad)
    n_dates = len(marked_spot_rates)
    a = 5 * np.ones(n_dates) / 100
    b = 0.005 * np.ones(n_dates)
    parameters = a
    bdt_lattice = black_derman_toy_lattice(np.array([a, b]), n_dates=n_dates)

    loss = black_derman_toy_loss(
        np.array([a, b]), marked_spot_rates, a_fixed=None, b_fixed=None
    )
    print(loss)
    calib = black_derman_toy_calibration(marked_spot_rates, b_fixed=b)
    # x_opt = calib.x.reshape((2, -1))
    # print(calib.x)
    print(calib)
    x_opt = np.array([calib.x, b])
    a, b = x_opt
    print(x_opt)
    optimal_spot_rates = compute_black_derman_toy_spot_rates(n_dates, x_opt)
    # # bdt = black_derman_toy_lattice(x_opt, n_dates)
    print(optimal_spot_rates.round(4))
    print(marked_spot_rates)
    # loss = black_derman_toy_loss(
    #     np.array([a, b]), marked_spot_rates, a_fixed=None, b_fixed=None
    # )
    # print(loss)
    # print(bdt[:5, :5])
    # print(x_opt)


def f(b):
    r00 = 0.06
    r10 = 0.054
    z = 88.64
    notional = 100
    r11 = r10 * np.exp(b)
    q = 0.5
    return 1 / (1 + r00) * (q * notional / (1 + r11) + q * notional / (1 + r10)) - z


def assignment():
    result = so.root_scalar(f, x0=0.005)
    print(np.round(result.root, 4))


if __name__ == "__main__":
    # main()
    assignment()
