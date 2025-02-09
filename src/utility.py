import numpy as np
from typing import Union


def logit(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert probability to logit with numerical stability."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -100, 100)))