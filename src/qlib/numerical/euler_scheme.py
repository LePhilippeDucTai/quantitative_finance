import numpy as np


def euler_discretization(  # noqa: PLR0913
    mu: callable,
    sigma: callable,
    t: np.ndarray,
    xt: np.ndarray,
    n_t: int,
    g: np.ndarray,
    dt: float,
    *model_args: tuple
) -> np.ndarray:
    for i in range(n_t - 1):
        t_i, x_i = t[i], xt[..., i]
        g_i = g[..., i]
        μ, σ = mu(t_i, x_i, *model_args), sigma(t_i, x_i, *model_args)
        xt[..., i + 1] = x_i + μ * dt + σ * g_i
    return xt
