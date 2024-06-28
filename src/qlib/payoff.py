import numpy as np

from qlib.brownian import Path


def call(paths: Path, r: float, k_strike: float, T: float):
    return np.exp(-r * T) * (paths.x[..., -1] - k_strike).clip(0)


def put(paths: Path, r: float, k_strike: float, T: float):
    return np.exp(-r * T) * (k_strike - paths.x[..., -1]).clip(0)


def asian_call(paths: Path, r: float, k_strike: float, T: float) -> np.ndarray:
    time_avg = np.mean(paths.x, axis=-1)
    return np.exp(-r * T) * (time_avg - k_strike).clip(0)


def barrier_payoff(
    x: np.ndarray, r: float, k_strike: float, T: float, indic: np.ndarray
):
    return np.exp(-r * T) * (x - k_strike).clip(0) * indic


def up_n_out_call(
    paths: Path, r: float, k_strike: float, barrier: float, T: float
) -> np.ndarray:
    indic_less_than_barrier = np.max(paths.x, axis=-1) <= barrier
    return barrier_payoff(paths.x[..., -1], r, k_strike, T, indic_less_than_barrier)


def up_n_in_call(
    paths: Path, r: float, k_strike: float, barrier: float, T: float
) -> np.ndarray:
    indic_max_stays_above_barrier = np.max(paths.x, axis=-1) >= barrier
    return barrier_payoff(
        paths.x[..., -1], r, k_strike, T, indic_max_stays_above_barrier
    )


def down_n_out_call(
    paths: Path, r: float, k_strike: float, barrier: float, T: float
) -> np.ndarray:
    indic_greater_than_barrier = np.min(paths.x, axis=-1) >= barrier
    return barrier_payoff(paths.x[..., -1], r, k_strike, T, indic_greater_than_barrier)


def down_n_in_call(
    paths: Path, r: float, k_strike: float, barrier: float, T: float
) -> np.ndarray:
    indic_min_stays_under_barrier = np.min(paths.x, axis=-1) <= barrier
    return barrier_payoff(
        paths.x[..., -1], r, k_strike, T, indic_min_stays_under_barrier
    )


def digital_option(paths: Path, r: float, k_strike: float, T: float):
    return np.exp(-r * T) * (paths.x[..., -1] > k_strike).astype(float)
