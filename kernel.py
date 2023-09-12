import numpy as np
import math


def rational_quadratic(src: np.ndarray, _lookback: int, _relative_weight: float, start_at_bar: int):
    _current_weight = 0.
    _cumulative_weight = 0.
    _t_val = math.pow(_lookback, 2) * 2 * _relative_weight
    for i in range(start_at_bar + 1):
        y = src[-i-1]
        w = math.pow(1 + (math.pow(i, 2) / _t_val), -_relative_weight)
        _current_weight += y * w
        _cumulative_weight += w
    return _current_weight / _cumulative_weight


def gaussian(src: np.ndarray, _lookback: int, start_at_bar: int):
    _current_weight = 0.
    _cumulative_weight = 0.
    _t_val = 2 * math.pow(_lookback, 2)
    for i in range(start_at_bar + 1):
        y = src[-i-1]
        w = math.exp(-math.pow(i, 2) / _t_val)
        _current_weight += y * w
        _cumulative_weight += w
    return _current_weight / _cumulative_weight
