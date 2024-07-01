"""Definition of all kinds of payoffs."""

import copy
from dataclasses import dataclass

import numpy as np
from qlib.constant_parameters import DEFAULT_SEED_SEQ, DELTA_EPSILON
from qlib.models.brownian import Path
from qlib.numerical.euler_scheme import ComputationKind
from qlib.traits import Model


@dataclass
class DerivativeParameters:
    maturity: float


@dataclass
class PricingData:
    price: float
    price_std: float
    delta: float
    gamma: float
    vega: float


class Derivative:
    def __init__(self, model: Model, parameters: DerivativeParameters):
        self.model = model
        self.parameters = parameters

    def generate_paths(
        self, kind: ComputationKind, generator: np.random.Generator
    ) -> Path:
        maturity = self.parameters.maturity
        match kind:
            case ComputationKind.EULER:
                return self.model.mc_euler(maturity, generator=generator)
            case ComputationKind.TERMINAL:
                return self.model.mc_terminal(maturity, generator=generator)
            case ComputationKind.EXACT:
                return self.model.mc_exact(
                    time_to_maturity=maturity, generator=generator
                )

    def npv(
        self,
        seed_seq: np.random.SeedSequence = DEFAULT_SEED_SEQ,
        kind: ComputationKind = ComputationKind.EULER,
    ):
        generator = np.random.default_rng(seed_seq)
        maturity = self.parameters.maturity
        df = self.model.term_structure.discount_factor(maturity)
        sample_paths = self.generate_paths(kind, generator)
        h = df * self.payoff(sample_paths)
        price = np.mean(h)
        std = np.std(h)
        return price, std

    def pricing(self, seed_seq: np.random.SeedSequence):
        price, std = self.npv(seed_seq)
        greeks = Sensitivities(self)
        delta, gamma = greeks.delta_gamma(seed_seq, ref=price)
        vega = greeks.vega(seed_seq, ref=price)
        return PricingData(price, std, delta, gamma, vega)


class Sensitivities:
    def __init__(self, derivative: Derivative):
        self.derivative = derivative

    def variate_model_parameters(
        self, model_parameter_name: str, eps: float
    ) -> Derivative:
        ds = copy.deepcopy(self.derivative)
        initial = ds.model.model_parameters.__getattribute__(model_parameter_name)
        next = initial + eps
        ds.model.model_parameters.__setattr__(model_parameter_name, next)
        return ds

    def delta_gamma(
        self, seed_seq: np.random.SeedSequence, ref=None, eps=DELTA_EPSILON
    ):
        d1 = self.variate_model_parameters("x0", eps)
        d2 = self.variate_model_parameters("x0", -eps)
        if ref is None:
            ref, _ = self.derivative.npv(seed_seq)
        (left, _), (right, _) = d1.npv(seed_seq), d2.npv(seed_seq)
        gamma = (left - 2 * ref + right) / eps**2
        delta = (left - right) / (2 * eps)
        return delta, gamma

    def vega(self, seed_seq: np.random.SeedSequence, ref=None, eps=DELTA_EPSILON):
        d = self.variate_model_parameters("sigma", eps)
        if ref is None:
            ref, _ = self.derivative.npv(seed_seq)
        value, _ = d.npv(seed_seq)
        return (value - ref) / eps


@dataclass
class EuropeanOptionParameters(DerivativeParameters):
    strike_k: float
    maturity: float


class EuropeanCallOption(Derivative):
    def __init__(self, model: Model, parameters: EuropeanOptionParameters):
        super().__init__(model, parameters)

    def payoff(self, sample_paths: Path):
        s = sample_paths.x[..., -1]
        k = self.parameters.strike_k
        return (s - k).clip(0)


class EuropeanPutOption(Derivative):
    def __init__(self, model, parameters):
        super().__init__(model, parameters)

    def payoff(self, sample_paths: Path):
        s = sample_paths.x[..., -1]
        k = self.parameters.strike_k
        return (k - s).clip(0)


class AsianCallOption(Derivative):
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
