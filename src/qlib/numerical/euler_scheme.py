from enum import Enum, auto

import numpy as np


class ComputationKind(Enum):
    DET = auto()
    EULER = auto()
    EULER_JIT = auto()
    EXACT = auto()
    MILSTEIN = auto()
    TERMINAL = auto()


def euler_discretization(  # noqa: PLR0913
    mu: callable,
    sigma: callable,
    t: np.ndarray,
    xt: np.ndarray,
    n_t: int,
    g: np.ndarray,
    dt: float,
) -> np.ndarray:
    for i in range(n_t - 1):
        t_i, x_i = t[i], xt[..., i]
        g_i = g[..., i]
        μ, σ = mu(t_i, x_i), sigma(t_i, x_i)
        xt[..., i + 1] = x_i + μ * dt + σ * g_i
    return xt


def milstein_discretization(
    mu: callable,
    sigma: callable,
    sigma_derivative: callable,
    t: np.ndarray,
    xt: np.ndarray,
    n_t: int,
    g: np.ndarray,
    dt: float,
) -> np.ndarray:
    for i in range(n_t - 1):
        t_i, x_i = t[i], xt[..., i]
        g_i = g[..., i]
        μ, σ = mu(t_i, x_i), sigma(t_i, x_i)
        sd = sigma_derivative(t_i, x_i)
        q = 0.5 * sd**2 * dt * (g_i**2 - 1)
        xt[..., i + 1] = x_i + μ * dt + σ * g_i + q
    return xt


def first_order_derivative(h, f1, fm1):
    return (f1 - fm1) / (2 * h)


def second_order_derivative(h: float, f2: float, f1, f0, fm1, fm2):
    return (-f2 + 16 * f1 - 30 * f0 + 16 * fm1 - fm2) / (12 * h**2)
