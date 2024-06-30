"""Definition of all kinds of payoffs."""

from dataclasses import dataclass

import numpy as np
from qlib.models.brownian import Path
from qlib.numerical.euler_scheme import ComputationKind
from qlib.traits import Model


@dataclass
class DerivativeParameters:
    maturity: float


class Derivative:
    def __init__(self, model: Model, parameters: DerivativeParameters):
        self.model = model
        self.parameters = parameters

    def npv(self, kind: ComputationKind):
        pass


@dataclass
class EuropeanOptionParameters(DerivativeParameters):
    strike_k: float
    maturity: float


class EuropeanOption(Derivative):
    def __init__(self, model: Model, parameters: EuropeanOptionParameters):
        super().__init__(model, parameters)
        self.parameters = parameters

    def generate_paths(self, kind: ComputationKind) -> Path:
        maturity = self.parameters.maturity
        match kind:
            case ComputationKind.EULER:
                return self.model.mc_euler(maturity)
            case ComputationKind.TERMINAL:
                return self.model.mc_terminal(maturity)
            case ComputationKind.EXACT:
                return self.model.mc_exact(maturity)

    def payoff(self, sample_paths: Path) -> np.ndarray:
        return 0

    def npv(self, kind: ComputationKind):
        maturity = self.parameters.maturity
        df = self.model.term_structure.discount_factor(maturity)
        sample_paths = self.generate_paths(kind)
        h = df * self.payoff(sample_paths)
        return np.mean(h)


class EuropeanCallOption(EuropeanOption):
    def __init__(self, model: Model, parameters: EuropeanOptionParameters):
        super().__init__(model, parameters)

    def payoff(self, sample_paths: Path):
        s = sample_paths.x[..., -1]
        k = self.parameters.strike_k
        return (s - k).clip(0)


class EuropeanPutOption(EuropeanOption):
    def __init__(self, model, parameters):
        super().__init__(model, parameters)

    def payoff(self, sample_paths: Path):
        s = sample_paths.x[..., -1]
        k = self.parameters.strike_k
        return (k - s).clip(0)


class AsianCallOption(EuropeanOption):
    def __init__(self, model, parameters):
        super().__init__(model, parameters)

    def payoff(self, sample_paths: Path):
        integral = (sample_paths.x).mean(axis=-1)
        k = self.parameters.strike_k
        return (integral - k).clip(0)


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
