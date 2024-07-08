import numba
import numpy as np


def bs_mu_jit(r: float) -> float:
    """Black-Scholes constant drift."""

    @numba.jit(nopython=True, cache=True)
    def f(t: float, x: float):
        return r * x

    return f


def bs_sigma_jit(sig: float) -> float:
    """Black-Scholes constant sigma."""

    @numba.jit(nopython=True, cache=True)
    def f(t: float, x: float):
        return sig * x

    return f


@numba.njit
def bs_mu_njit(t: float, x, r: float, sigma: float):
    return r * x


@numba.njit
def bs_sigma_njit(t: float, xt: float, r: float, sigma: float):
    return sigma * xt


@numba.njit
def dummy(t: float, x: float):
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
) -> np.ndarray:
    """Euler schema discretization for Ito Processes SDE."""
    for i in range(n_t - 1):
        t_i, x_i = t[i], xt[..., i]
        g_i = g[..., i]
        μ, σ = mu_jit(t_i, x_i), sigma_jit(t_i, x_i)
        xt[..., i + 1] = x_i + μ * dt + σ * g_i
    return xt


# Compiling the functions
a = bs_mu_jit(3)
b = bs_sigma_jit(0)
_ = euler_discretization_jit(a, b, np.arange(1), np.arange(1), 1, np.arange(1), 1)
