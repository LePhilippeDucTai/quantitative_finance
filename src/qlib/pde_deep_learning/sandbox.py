import numpy as np
import tensorflow as tf
from qlib.utils.timing import time_it


def neural_network():
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(20, activation="tanh"),
            tf.keras.layers.Dense(20, activation="tanh"),
            tf.keras.layers.Dense(1),
        ]
    )


def bs_pde(x, u, u_t, u_x, u_xx, r, sigma):
    return u_t + u_x * r * x + 0.5 * u_xx * sigma**2 * x**2 - r * u


def space_time_grid(x_lim, t_lim, n_x, n_t):
    x, t = np.linspace(0, x_lim, n_x), np.linspace(0, t_lim, n_t)
    X, T = np.meshgrid(x, t)


@time_it
def main():
    # x = tf.Variable(tf.Tensor([0, 1, 3, 4]))
    x = tf.Variable(5.0)
    with tf.GradientTape() as tape:
        y = x**3 + 0.3 * x**2 - 1

    dy_dx = tape.gradient(y, x)
    print(dy_dx.numpy())

    model = neural_network()
    print(model.summary())


if __name__ == "__main__":
    main()
