from typing import Any

import numpy as np
from qlib.models.brownian import Path


def npv(payoff: callable, paths: Path, *args: tuple, **kwargs: dict[str, Any]):
    """Monte-Carlo expectation."""
    return np.mean(payoff(paths, *args, **kwargs))
