"""Definition of all kinds of payoffs."""

import numpy as np

from qlib.models.brownian import Path


def call(s: np.ndarray, r: float, k_strike: float, tmt: float):
    return np.exp(-r * tmt) * (s - k_strike).clip(0)


def put(s: np.ndarray, r: float, k_strike: float, tmt: float):
    return np.exp(-r * tmt) * (k_strike - s).clip(0)


def asian_call(paths: Path, r: float, k_strike: float, tmt: float) -> np.ndarray:
    time_avg = np.mean(paths.x, axis=-1)
    return np.exp(-r * tmt) * (time_avg - k_strike).clip(0)


def barrier_payoff(
    x: np.ndarray,
    r: float,
    k_strike: float,
    tmt: float,
    indic: np.ndarray,
):
    return np.exp(-r * tmt) * (x - k_strike).clip(0) * indic


def up_n_out_call(
    paths: Path, r: float, k_strike: float, barrier: float, tmt: float
) -> np.ndarray:
    indic_less_than_barrier = np.max(paths.x, axis=-1) <= barrier
    return barrier_payoff(paths.x[..., -1], r, k_strike, tmt, indic_less_than_barrier)


def up_n_in_call(
    paths: Path, r: float, k_strike: float, barrier: float, tmt: float
) -> np.ndarray:
    indic_max_stays_above_barrier = np.max(paths.x, axis=-1) >= barrier
    return barrier_payoff(
        paths.x[..., -1], r, k_strike, tmt, indic_max_stays_above_barrier
    )


def down_n_out_call(
    paths: Path, r: float, k_strike: float, barrier: float, tmt: float
) -> np.ndarray:
    indic_greater_than_barrier = np.min(paths.x, axis=-1) >= barrier
    return barrier_payoff(
        paths.x[..., -1], r, k_strike, tmt, indic_greater_than_barrier
    )


def down_n_in_call(
    paths: Path, r: float, k_strike: float, barrier: float, tmt: float
) -> np.ndarray:
    indic_min_stays_under_barrier = np.min(paths.x, axis=-1) <= barrier
    return barrier_payoff(
        paths.x[..., -1], r, k_strike, tmt, indic_min_stays_under_barrier
    )


def digital_option(paths: Path, r: float, k_strike: float, tmt: float):
    return np.exp(-r * tmt) * (paths.x[..., -1] > k_strike).astype(float)
