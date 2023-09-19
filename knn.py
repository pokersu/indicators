import warnings
import numpy as np
import pandas as pd
import math
from . import kernel
from typing import List
from .indicators import cci, wma, sma, rsi, typical_price
from pandas.core.base import PandasObject

warnings.simplefilter(action="ignore", category=RuntimeWarning)


def nz(series, nan: np.float64 = np.float64(0.)):
    return series.replace(np.nan, nan)


def normalize(series: pd.Series, min_val: np.float64, max_val: np.float64):
    _historic_min = np.float64(10e10)
    _historic_max = np.float64(-10e10)

    def k(val):
        nonlocal _historic_min, _historic_max
        _historic_min = _historic_min if pd.isna(val) else min(val, _historic_min)
        _historic_max = _historic_max if pd.isna(val) else max(val, _historic_max)
        _distance = _historic_max - _historic_min
        return min_val + (max_val - min_val) * (val - _historic_min) / max(_distance, np.float64(10e-10))

    return series.apply(k)


def rescale(series, old_min: np.float64, old_max: np.float64, new_min: np.float64, new_max: np.float64):
    return new_min + (new_max - new_min) * (series - old_min) / max(old_max - old_min, 10e-10)


def n_cci(bars: pd.DataFrame, cci_window: int = 20, ema_window: int = 1):
    # todo: 这里与 trading view 中 ta.cci 计算方式不一样
    series = cci(bars, cci_window)
    series = wma(series, ema_window)
    return normalize(series, np.float64(0.), np.float64(1.))


def n_wt(bars: pd.DataFrame, ema_window_1: int = 10, ema_window_2: int = 11):
    series = typical_price(bars)
    ema1 = wma(series, ema_window_1)
    ema2 = wma(pd.Series(series - ema1).abs(), ema_window_1)
    ci = (series - ema1) / (0.015 * ema2)
    wt1 = wma(ci, ema_window_2)
    wt2 = sma(wt1, 4)
    return normalize(wt1 - wt2, np.float64(0.), np.float64(1.))


def n_rsi(bars: pd.DataFrame, rsi_window: int = 14, ema_window: int = 1):
    series = bars['close']
    series = rsi(series, rsi_window)
    series = wma(series, ema_window)
    return rescale(series, np.float64(0.), np.float64(100.), np.float64(0.), np.float64(1.))


def s_adx(data: pd.DataFrame, length: int):
    bars = data.copy(deep=True)
    bars['tr_smooth'], bars['smooth_direct_movement'], bars['smooth_neg_movement'], bars['dxs'] = 0., 0., 0., 0.
    bars.index = range(len(bars))
    cols = bars.columns
    alpha = 1. / length

    def conv(x):
        pre_idx, idx = x.index
        pre_row, row = bars.iloc[pre_idx], bars.iloc[idx]
        tr = max(max(row['high'] - row['low'], abs(row['high'] - pre_row['close'])), abs(row['low'] - pre_row['close']))
        directional_movement_plus = max(row['high'] - pre_row['high'], 0.) if row['high'] - pre_row['high'] > pre_row[
            'low'] - row['low'] else 0
        neg_movement = max(pre_row['low'] - row['low'], 0.) if pre_row['low'] - row['low'] > row['high'] - pre_row[
            'high'] else 0.
        tr_smooth_val = pre_row['tr_smooth'] - pre_row['tr_smooth'] / length + tr
        smooth_direct_movement_val = pre_row['smooth_direct_movement'] - pre_row[
            'smooth_direct_movement'] / length + directional_movement_plus
        smooth_neg_movement_val = pre_row['smooth_neg_movement'] - pre_row[
            'smooth_neg_movement'] / length + neg_movement
        bars.iat[idx, cols.get_loc('tr_smooth')] = tr_smooth_val
        bars.iat[idx, cols.get_loc('smooth_direct_movement')] = smooth_direct_movement_val
        bars.iat[idx, cols.get_loc('smooth_neg_movement')] = smooth_neg_movement_val

        di_positive = smooth_direct_movement_val / tr_smooth_val * 100.
        di_negative = smooth_neg_movement_val / tr_smooth_val * 100.
        dx = abs(di_positive - di_negative) / (di_positive + di_negative) * 100.
        _adx = alpha * dx + (1. - alpha) * pre_row['dxs']
        bars.iat[idx, cols.get_loc('dxs')] = _adx
        return _adx

    adx = bars['open'].rolling(2).apply(conv)
    adx.index = data.index
    return adx


def n_adx(data: pd.DataFrame, length: int):
    adx = s_adx(data, length)
    return rescale(adx, np.float64(0.), np.float64(100.), np.float64(0.), np.float64(1.))


def regime(data: pd.DataFrame):
    bars = data.copy(deep=True)
    bars['ohlc4'] = (bars['open'] + bars['high'] + bars['low'] + bars['close']) / 4.
    bars['value1'], bars['value2'], bars['klmf'] = 0., 0., 0.
    bars.index = range(len(bars))
    cols = bars.columns

    def conv(x):
        pre_idx, idx = x.index
        pre_row, row = bars.iloc[pre_idx], bars.iloc[idx]
        val1 = 0.2 * (row['ohlc4'] - pre_row['ohlc4']) + 0.8 * pre_row['value1']
        val2 = 0.1 * (row['high'] - row['low']) + 0.8 * pre_row['value2']
        bars.iat[idx, cols.get_loc('value1')] = val1
        bars.iat[idx, cols.get_loc('value2')] = val2
        omega = abs(val1 / val2)
        alpha = (-math.pow(omega, 2) + math.sqrt(math.pow(omega, 4) + 16 * math.pow(omega, 2))) / 8
        klmf = alpha * row['ohlc4'] + (1 - alpha) * pre_row['klmf']
        bars.iat[idx, cols.get_loc('klmf')] = klmf
        abs_curve_slope = abs(klmf - pre_row['klmf'])
        return abs_curve_slope

    abs_curve_slope_series = bars['open'].rolling(2).apply(conv)
    abs_curve_slope_series.index = data.index
    exponential_average_abs_curve_slope = wma(abs_curve_slope_series, 200) * 1.0
    return (abs_curve_slope_series - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope


def train_labeling(bars: pd.DataFrame, features: list[str]):
    data_len = len(bars)

    def conv(c):
        c4, c0 = c[0], c[3]
        if c4 < c0:
            return -1
        elif c4 > c0:
            return 1
        else:
            return 0

    def label(row):
        index = row.name
        historic = bars[:index].iloc[:-1]
        his_data_len = len(historic)
        if his_data_len < 3000 or data_len - his_data_len > 300:
            return np.nan

        point = row[features].values.astype(np.float64)
        points = historic[features].values.astype(np.float64)
        distance_series = pd.Series(np.sum(np.log(1 + np.abs(points - point)), axis=1), index=historic.index)

        last_distance = -1.0
        distances = []
        predictions = []
        c = 0
        for index, d in distance_series.items():
            if d >= last_distance and c % 4 == 0:
                last_distance = d
                distances.append(d)
                predictions.append(historic.loc[index]['trend'])
                if len(distances) >= 8:
                    last_distance = distance_series[6]
                    distances.pop(0)
                    predictions.pop(0)
            c += 1
        return sum(predictions)

    bars['trend'] = bars['close'].rolling(window=4).apply(conv)
    return bars.apply(label, axis=1)


# def distance(bars: pd.DataFrame, row: pd.Series, features: List[str]):
#     points = bars[features].values.astype(np.float64)
#     point = row[features].values.astype(np.float64)
#     return np.sum(np.log(1 + np.abs(points - point)), axis=1)


def rational_quadratic(bars: pd.DataFrame, _lookback: int, _relative_weight: float, start_at_bar: int):
    idx = 0
    series = bars['close']

    def k(_):
        nonlocal idx
        if idx <= 25:
            idx = idx + 1
            return np.nan
        data = series[:idx + 1]
        idx = idx + 1
        return kernel.rational_quadratic(data, _lookback, _relative_weight, start_at_bar)

    return series.apply(k)


def gaussian(bars: pd.DataFrame, _lookback: int, start_at_bar: int):
    idx = 0
    series = bars['close']

    def k(_):
        nonlocal idx
        if idx <= 25:
            idx = idx + 1
            return np.nan
        data = series[:idx + 1]
        idx = idx + 1
        return kernel.gaussian(data, _lookback, start_at_bar)

    return series.apply(k)


PandasObject.nz = nz
# PandasObject.distance = distance
PandasObject.normalize = normalize
PandasObject.rescale = rescale
PandasObject.n_cci = n_cci
PandasObject.n_wt = n_wt
PandasObject.n_rsi = n_rsi
PandasObject.n_adx = n_adx
PandasObject.s_adx = s_adx
PandasObject.train_labeling = train_labeling
PandasObject.regime = regime
PandasObject.rational_quadratic = rational_quadratic
PandasObject.gaussian = gaussian
