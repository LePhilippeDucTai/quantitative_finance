"""List of all constant parameters."""

import numpy as np

# Gives by default for each time interval [t, t + 1], 2 ** 10 = 1024 points
N_DYADIC = 10
N_MC = 10000

DEFAULT_RNG = np.random.default_rng()
DEFAULT_SEED_SEQ = np.random.SeedSequence(0)
DELTA_EPSILON = 0.01
EPSILON = np.finfo(float).eps
ETA = EPSILON ** (1 / 3)
