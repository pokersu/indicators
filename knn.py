import warnings
import numpy as np
import pandas as pd
import math
from typing import List
from indicators import cci, wma, sma, rsi
from pandas.core.base import PandasObject

warnings.simplefilter(action="ignore", category=RuntimeWarning)


def nz(series, nan: np.float64 = np.float64(0.)):
    return series.replace(np.nan, nan)


def normalize(series, min_val: np.float64, max_val: np.float64):
    _historic_min = np.float64(10e10)
    _historic_max = np.float64(-10e10)

    _historic_min = min(nz(series, nan=_historic_min), _historic_min)
    _historic_max = max(nz(series, nan=_historic_max), _historic_max)
    return min_val + (max_val - min_val) * (series - _historic_min) / max(_historic_max - _historic_min,
                                                                          np.float64(10e-10))


def rescale(series, old_min: np.float64, old_max: np.float64, new_min: np.float64, new_max: np.float64):
    new_min + (new_max - new_min) * (series - old_min) / max(old_max - old_min, 10e-10)


def n_cci(series, cci_window: int, ema_window: int):
    series = cci(series, cci_window)
    series = wma(series, ema_window)
    return normalize(series, np.float64(0.), np.float64(100.))


def n_wt(series, ema_window_1: int, ema_window_2: int):
    ema1 = wma(series, ema_window_1)
    ema2 = wma(pd.Series(series - ema1).abs(), ema_window_1)
    ci = (series - ema1) / (0.015 * ema2)
    wt1 = wma(ci, ema_window_2)
    wt2 = sma(wt1, 4)
    return normalize(wt1 - wt2, np.float64(0.), np.float64(1.))


def n_rsi(series, rsi_window: int, ema_window: int):
    series = rsi(series, rsi_window)
    series = wma(series, ema_window)
    return rescale(series, np.float64(0.), np.float64(100.), np.float64(0.), np.float64(1.))


def n_adx(bars: pd.DataFrame, length: int):
    tr_smooth = pd.Series([0.0] * bars.size)
    smooth_direct_movement = pd.Series([0.0] * bars.size)
    smooth_neg_movement = pd.Series([0.0] * bars.size)
    dxs = pd.Series([0.0] * bars.size)
    alpha = 1. / length

    def _nz(x: float):
        return 0. if x == np.nan else x

    def k(row):
        idx = row.name
        pre_idx = idx - 1
        high, low, close = row['high'], row['low'], row['close']
        pre_high, pre_low = bars.iloc[pre_idx]['high'], bars.iloc[pre_idx]['low']
        pre_close, pre_tr_smooth = bars.iloc[pre_idx]['close'], tr_smooth[pre_idx]
        pre_smooth_directional_movement_plus = smooth_direct_movement[pre_idx]
        pre_tr_smooth_neg_movement = smooth_neg_movement[pre_idx]
        pre_dx = dxs[pre_idx]
        pre_high, pre_low, pre_close = _nz(pre_high), _nz(pre_low), _nz(pre_close)
        pre_tr_smooth = _nz(pre_tr_smooth)
        pre_smooth_directional_movement_plus = _nz(pre_smooth_directional_movement_plus)
        pre_tr_smooth_neg_movement = _nz(pre_tr_smooth_neg_movement)
        tr = max(max(high - low, abs(high - pre_close)), abs(low - pre_close))
        directional_movement_plus = max(high - pre_high, 0.) if high - pre_high > pre_low - low else 0
        neg_movement = max(pre_low - low, 0.) if pre_low - low > high - pre_high else 0.
        tr_smooth[idx] = pre_tr_smooth - pre_tr_smooth / length + tr
        smooth_direct_movement[
            idx] = pre_smooth_directional_movement_plus - pre_smooth_directional_movement_plus / length + directional_movement_plus
        smooth_neg_movement[idx] = pre_tr_smooth_neg_movement - pre_tr_smooth_neg_movement / length + neg_movement
        di_positive = smooth_direct_movement[idx] / tr_smooth[idx] * 100.
        di_negative = smooth_neg_movement[idx] / tr_smooth[idx] * 100.
        dx = abs(di_positive - di_negative) / (di_positive + di_negative) * 100.
        return dx * alpha + (1. - alpha) * pre_dx

    adx = bars.apply(k, axis=1)
    return rescale(adx, np.float64(0.), np.float64(100.), np.float64(0.), np.float64(1.))


def normalized_slope_decline(bars: pd.DataFrame):
    value1 = pd.Series([0.0] * bars.size)
    value2 = pd.Series([0.0] * bars.size)
    klmf = pd.Series([0.0] * bars.size)

    def _nz(x: float):
        return 0. if x == np.nan else x

    def k(row):
        idx = row.name
        pre_idx = idx - 1
        high, low, close = row['high'], row['low'], row['close']
        pre_high, pre_low = bars.iloc[pre_idx]['high'], bars.iloc[pre_idx]['low']
        pre_close = bars.iloc[pre_idx]['close']
        pre_klmf = klmf[pre_idx]
        pre_high, pre_low, pre_close = _nz(pre_high), _nz(pre_low), _nz(pre_close)
        pre_value1, pre_value2 = value1[pre_idx], value2[pre_idx]
        value1[idx] = 0.2 * (close - pre_close) + 0.8 * _nz(pre_value1)
        value2[idx] = 0.1 * (high - low) + 0.8 * _nz(pre_value2)
        omega = abs(value1[idx] / value2[idx])
        alpha = (-math.pow(omega, 2) + math.sqrt(math.pow(omega, 4) + 16 * math.pow(omega, 2))) / 8
        klmf[idx] = alpha * close + (1 - alpha) * pre_klmf
        abs_curve_slope = abs(klmf[idx] - pre_klmf)
        return abs_curve_slope

    abs_curve_slope_series = bars.apply(k, axis=1)
    exponential_average_abs_curve_slope = wma(abs_curve_slope_series, 200) * 1.0
    return (abs_curve_slope_series - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope


def distance(bars: pd.DataFrame, row: pd.Series, features: List[str]):
    points = bars[features].values.astype(np.float64)
    point = row[features].values.astype(np.float64)
    return np.sum(np.log(1 + np.abs(points - point)), axis=1)

PandasObject.nz = nz
PandasObject.distance = distance
PandasObject.normalize = normalize
PandasObject.rescale = rescale
PandasObject.n_cci = n_cci
PandasObject.n_wt = n_wt
PandasObject.n_rsi = n_rsi
PandasObject.n_adx = n_adx
PandasObject.normalized_slope_decline = normalized_slope_decline
