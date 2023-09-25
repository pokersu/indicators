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
    max_values = series.expanding().max()
    min_values = series.expanding().min()
    distances = max_values - min_values
    distances = distances.apply(lambda d: max(d, np.float64(10e-10)))
    calc_df = pd.concat([series, min_values, distances], axis=1)
    return min_val + (max_val - min_val) * (calc_df[0] - calc_df[1]) / calc_df[2]


def rescale(series, old_min: np.float64, old_max: np.float64, new_min: np.float64, new_max: np.float64):
    return new_min + (new_max - new_min) * (series - old_min) / max(old_max - old_min, 10e-10)


def n_cci(bars: pd.DataFrame, cci_window: int = 20, ema_window: int = 1):
    price = bars['close']
    ma = sma(price, cci_window)
    calc_s = pd.concat([price, ma], axis=1, keys=['p', 'm'])

    def k(row):
        idx = row.name
        historic_df = calc_s[:idx]
        if len(historic_df) <= cci_window:
            return np.nan
        temp_df = historic_df[-cci_window:]
        mean = temp_df['m'][-1]
        return sum(abs(temp_df['p'] - mean)) / cci_window

    md = calc_s.apply(k, axis=1)
    series = (price - ma) / (.015 * md)
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
    def conv(c):
        c4, c0 = c[0], c[4]
        if c4 < c0:
            return -1
        elif c4 > c0:
            return 1
        else:
            return 0

    distances = []
    predictions = []

    def distance_conv(window):
        idx = window.index
        rows = bars.loc[idx]
        point = rows.iloc[-1][features].values.astype(np.float64)
        points = rows[features].values.astype(np.float64)
        distances_series = pd.Series(np.sum(np.log(1 + np.abs(points - point)), axis=1), index=idx)
        # distances_series = distances_series.iloc[:2000]
        last_distance = -1.0

        for i, (index, d) in enumerate(distances_series.items()):
            if d >= last_distance and i % 4 == 0:
                last_distance = d
                distances.append(d)
                predictions.append(rows.loc[index]['trend'])
                if len(predictions) > 8:
                    last_distance = distances[6]
                    distances.pop(0)
                    predictions.pop(0)
        return sum(predictions)

    bars['trend'] = bars['close'].rolling(window=5).apply(conv)
    return bars['close'].rolling(window=2000).apply(distance_conv)


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


def render_signal(bars: pd.DataFrame):
    def k(row):
        if row['train_label'] > 0 and row['filter_all']:
            return 1
        elif row['train_label'] < 0 and row['filter_all']:
            return -1
        else:
            return np.nan

    signals = bars.apply(k, axis=1)
    return signals.fillna(method='ffill')


def render_signal_held(bars: pd.DataFrame):
    series = bars['signal']
    return series.groupby(series.ne(series.shift()).cumsum()).cumcount() + 1


def render_signal_flip(bars: pd.DataFrame):
    signals = bars['signal']
    s_shift = signals.shift(1)
    return s_shift != signals


def render_ema_trend(bars: pd.DataFrame, ema_window: int):
    ema_s = wma(bars['close'], ema_window)
    return pd.Series(np.where(bars['close'] > ema_s, 1, np.where(bars['close'] < ema_s, -1, 0)), index=bars.index)


def render_sma_trend(bars: pd.DataFrame, sma_window: int):
    sma_s = sma(bars['close'], sma_window)
    return pd.Series(np.where(bars['close'] > sma_s, 1, np.where(bars['close'] < sma_s, -1, 0)), index=bars.index)


def render_kernel_trend(bars: pd.DataFrame, smooth=False):
    if smooth:
        return pd.Series(np.where(bars['yhat2'] >= bars['yhat1'], 1, -1), index=bars.index)
    else:
        s = bars['yhat1']
        s_ = s.shift(1)
        return (s > s_).astype(int) - (s < s_).astype(int)


def render_long(bars: pd.DataFrame):
    condition = np.where((bars['signal'] > 0)
                         & (bars['signal_flip'])
                         & (bars['ema_trend'] > 0)
                         & (bars['sma_trend'] > 0)
                         & (bars['kernel_trend'] > 0), True, False)
    # condition = np.where((bars['signal'] > 0), True, False)
    return pd.Series(condition, index=bars.index)


def render_short(bars: pd.DataFrame):
    condition = np.where((bars['signal'] < 0)
                         & (bars['signal_flip'])
                         & (bars['ema_trend'] < 0)
                         & (bars['sma_trend'] < 0)
                         & (bars['kernel_trend'] < 0), True, False)
    # condition = np.where((bars['signal'] < 0), True, False)
    return pd.Series(condition, index=bars.index)


PandasObject.nz = nz
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
PandasObject.render_signal = render_signal
PandasObject.render_signal_held = render_signal_held
PandasObject.render_signal_flip = render_signal_flip
PandasObject.render_ema_trend = render_ema_trend
PandasObject.render_sma_trend = render_sma_trend
PandasObject.render_kernel_trend = render_kernel_trend
PandasObject.render_long = render_long
PandasObject.render_short = render_short
