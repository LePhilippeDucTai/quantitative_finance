"""Definition of all kinds of payoffs."""

import copy
from dataclasses import dataclass

import numpy as np
from qlib.constant_parameters import DEFAULT_SEED_SEQ, DELTA_EPSILON, ETA
from qlib.models.brownian import Path
from qlib.numerical.euler_scheme import ComputationKind, first_order_derivative
from qlib.traits import Model


@dataclass
class DerivativeParameters:
    maturity: float


@dataclass
class PricingData:
    price: float
    price_std: float
    delta: float
    vega: float
    gamma: float = None


class Derivative:
    def __init__(self, model: Model, parameters: DerivativeParameters):
        self.model = model
        self.parameters = parameters

    def generate_paths(
        self, kind: ComputationKind, seed_seq: np.random.SeedSequence
    ) -> Path:
        maturity = self.parameters.maturity
        generator = np.random.default_rng(seed_seq)
        match kind:
            case ComputationKind.EULER:
                return self.model.mc_euler(maturity, generator=generator)
            case ComputationKind.EULER_JIT:
                return self.model.mc_euler_jit(maturity, generator=generator)
            case ComputationKind.TERMINAL:
                return self.model.mc_terminal(maturity, generator=generator)
            case ComputationKind.EXACT:
                return self.model.mc_exact(
                    time_to_maturity=maturity, generator=generator
                )
            case ComputationKind.MILSTEIN:
                return self.model.mc_milstein(maturity, generator=generator)

    def payoff(self, sample_paths):
        pass

    def npv(
        self,
        seed_seq: np.random.SeedSequence = DEFAULT_SEED_SEQ,
        kind: ComputationKind = ComputationKind.EULER,
        std: bool = False,
    ) -> float | tuple[float]:

        maturity = self.parameters.maturity
        df = self.model.term_structure.discount_factor(maturity)
        sample_paths = self.generate_paths(kind, seed_seq)
        h = df * self.payoff(sample_paths)
        price = np.mean(h)
        if std:
            std = np.std(h)
            return price, std
        return price

    def pricing(self, seed_seq: np.random.SeedSequence, kind: ComputationKind):
        price, std = self.npv(seed_seq, kind=kind, std=True)
        greeks = Sensitivities(self)
        delta = greeks.delta(seed_seq, kind=kind)
        gamma = greeks.gamma(seed_seq, kind=kind)
        vega = greeks.vega(seed_seq, kind=kind)
        return PricingData(
            price=price.real,
            price_std=std,
            delta=delta.real,
            vega=vega.real,
            gamma=gamma,
        )


class Sensitivities:
    def __init__(self, derivative: Derivative):
        self.derivative = derivative

    def variate_model_parameters(
        self, model_parameter_name: str, h: float
    ) -> Derivative:
        ds = copy.deepcopy(self.derivative)
        initial = ds.model.model_parameters.__getattribute__(model_parameter_name)
        next = initial + h
        ds.model.model_parameters.__setattr__(model_parameter_name, next)
        return ds

    def _h(self, model_parameter_name: str):
        initial = self.derivative.model.model_parameters.__getattribute__(
            model_parameter_name
        )
        return ETA * (1 + np.abs(initial))

    def delta(self, seed_seq: np.random.SeedSequence, kind: ComputationKind, dx=0.0):
        h = self._h("x0")
        fc = self.variate_model_parameters("x0", 1j * h + dx).npv(seed_seq, kind=kind)
        return fc.imag / h

    def gamma(
        self, seed_seq: np.random.SeedSequence, kind: ComputationKind, eps=DELTA_EPSILON
    ):
        d1 = self.delta(seed_seq, kind=kind, dx=eps)
        dm1 = self.delta(seed_seq, kind=kind, dx=-eps)
        return first_order_derivative(eps, d1, dm1)

    def vega(self, seed_seq: np.random.SeedSequence, kind: ComputationKind):
        h = self._h("x0")
        f1 = self.variate_model_parameters("sigma", h).npv(seed_seq, kind=kind)
        fm1 = self.variate_model_parameters("sigma", -h).npv(seed_seq, kind=kind)
        return first_order_derivative(h, f1, fm1)


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
