import numpy as np
import math


def rational_quadratic(src: np.ndarray, _lookback: int, _relative_weight: float, start_at_bar: int):
    _current_weight = 0.
    _cumulative_weight = 0.
    _size = src.size
    for i in _size + start_at_bar:
        y = src[i]
        _t_val = math.pow(i, 2) / (math.pow(_lookback, 2) * 2 * _relative_weight)
        w = math.pow(1 + _t_val, -_relative_weight)
        _current_weight += y * w
        _cumulative_weight += w
    return _current_weight / _cumulative_weight


def gaussian(src: np.ndarray, _lookback: int, start_at_bar: int):
    _current_weight = 0.
    _cumulative_weight = 0.
    _size = src.size
    for i in _size + start_at_bar:
        y = src[i]
        w = math.exp(-math.pow(i, 2) / (2 * math.pow(_lookback, 2)))
        _current_weight += y * w
        _cumulative_weight += w
    return _current_weight / _cumulative_weight
