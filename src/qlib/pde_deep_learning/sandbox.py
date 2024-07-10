import numpy as np
import tensorflow as tf
from qlib.utils.logger import logger


def neural_network():
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(10, activation="tanh"),
            tf.keras.layers.Dense(10, activation="tanh"),
            tf.keras.layers.Dense(1),
        ]
    )


def bs_pde(x, u, u_t, u_x, u_xx, r, sigma):
    return u_t + u_x * r * x + 0.5 * u_xx * sigma**2 * x**2 - r * u


def bs_terminal(t_x: np.ndarray, K: float):
    return (K - t_x[:, 1]).clip(0)


def cartesian_product(*arrays):
    mesh = np.meshgrid(*arrays)
    return np.vstack([u.flatten() for u in mesh]).transpose()


def loss(model, X_train, terminal_X_train, final_value, r, sigma):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X_train)
        u_pred = model(X_train)
        u_t = tape.gradient(u_pred, X_train)[:, 0]
        u_x = tape.gradient(u_pred, X_train)[:, 1]
    u_xx = tape.gradient(u_x, X_train)[:, 1]
    x = X_train[:, 1]
    f = bs_pde(x, u_pred, u_t, u_x, u_xx, r, sigma)
    mse = tf.reduce_mean(f**2)
    u_term = model(terminal_X_train)
    mse_term = tf.reduce_mean(tf.square(u_term - final_value))
    return mse + mse_term


def main():

    # s0 = 100
    K = 100
    r = 0.05
    sigma = 0.25
    n_x, n_t = 100, 100
    x_lim, t_lim = 200, 5
    x = np.linspace(0, x_lim, n_x)
    t = np.linspace(0, t_lim, n_t)

    training_data = cartesian_product(t, x)
    cond = training_data[:, 0] == t_lim
    terminal_training_data = training_data[cond]
    X_train = tf.convert_to_tensor(training_data[~cond], dtype=tf.float32)
    terminal_X_train = tf.convert_to_tensor(terminal_training_data, dtype=tf.float32)
    model = neural_network()
    u_final = tf.convert_to_tensor(
        bs_terminal(terminal_training_data, K).reshape(-1, 1), dtype=tf.float32
    )

    optimizer = tf.keras.optimizers.legacy.Adam()
    epochs = 500

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            current_loss = loss(model, X_train, terminal_X_train, u_final, r, sigma)
        grads = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 100 == 0:
            logger.info(f"Epoch : {epoch}, Loss : {current_loss.numpy()}")


if __name__ == "__main__":
    main()
