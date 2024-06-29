import numba
import numpy as np


@numba.njit
def bs_mu_jit(_: float, x: float, r: float, *args) -> float:
    """Black-Scholes constant drift."""
    return r * x


@numba.njit
def bs_sigma_jit(_: float, x: float, r: float, sig: float) -> float:
    """Black-Scholes constant sigma."""
    return sig * x


@numba.njit
def bs_mu_njit(t: float, x, r: float, sigma: float):
    return r * x


@numba.njit
def bs_sigma_njit(t: float, xt: float, r: float, sigma: float):
    return sigma * xt


@numba.njit
def dummy(t: float, x: float, *args):
    """Define a function for compilation."""
    return t * x


@numba.njit
def euler_discretization_jit(  # noqa: PLR0913
    mu_jit: callable,
    sigma_jit: callable,
    t: np.ndarray,
    xt: np.ndarray,
    n_t: int,
    g: np.ndarray,
    dt: float,
    *model_args: tuple
) -> np.ndarray:
    """Euler schema discretization for Ito Processes SDE."""
    for i in range(n_t - 1):
        t_i, x_i = t[i], xt[..., i]
        g_i = g[..., i]
        μ, σ = mu_jit(t_i, x_i, *model_args), sigma_jit(t_i, x_i, *model_args)
        xt[..., i + 1] = x_i + μ * dt + σ * g_i
    return xt


# Compiling the functions
_ = bs_mu_njit(0, 0, 0, 0)
_ = bs_sigma_njit(0, 0, 0, 0)
_ = euler_discretization_jit(
    dummy, dummy, np.arange(1), np.arange(1), 1, np.arange(1), 1, _
)
