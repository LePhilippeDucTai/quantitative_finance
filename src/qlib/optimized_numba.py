import numba
import numpy as np


@numba.njit
def bs_mu_jit(r: float, _: float, x: float) -> float:
    """Black-Scholes constant drift."""
    return r * x


@numba.njit
def bs_sigma_jit(sig: float, _: float, x: float) -> float:
    """Black-Scholes constant sigma."""
    return sig * x


@numba.njit
def dummy(t: float, x: float):
    """Define a function for compilation."""
    return t * x


@numba.njit
def euler_discretization(  # noqa: PLR0913
    mu_jit: callable,
    sigma_jit: callable,
    t: np.ndarray,
    xt: np.ndarray,
    n_t: int,
    g: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Euler schema discretization for Ito Processes SDE."""
    for i in range(n_t - 1):
        t_i, x_i = t[i], xt[..., i]
        g_i = g[..., i]
        μ, σ = mu_jit(t_i, x_i), sigma_jit(t_i, x_i)
        xt[..., i + 1] = x_i + μ * dt + σ * g_i
    return xt


# Compiling the functions
_ = bs_mu_jit(0, 0, 0)
_ = bs_sigma_jit(0, 0, 0)
_ = euler_discretization(dummy, dummy, np.arange(1), np.arange(1), 1, np.arange(1), 1)
