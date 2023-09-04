import numpy as np
import talib as ta


def nz(src: np.ndarray, nan: np.float64 = np.float64(0.)):
    return np.nan_to_num(src, nan=nan)


def normalize(src: np.ndarray, min_val: np.float64, max_val: np.float64):
    _historic_min = np.float64(10e10)
    _historic_max = np.float64(-10e10)

    _historic_min = min(nz(src, nan=_historic_min), _historic_min)
    _historic_max = max(nz(src, nan=_historic_max), _historic_max)
    return min_val + (max_val - min_val) * (src - _historic_min) / max(_historic_max - _historic_min,
                                                                       np.float64(10e-10))


def rescale(src: np.ndarray, old_min: np.float64, old_max: np.float64, new_min: np.float64, new_max: np.float64):
    new_min + (new_max - new_min) * (src - old_min) / max(old_max - old_min, 10e-10)


def n_rsi(src: np.ndarray, n1: int, n2: int):
    rsi = ta.RSI(src, timeperiod=n1)
    e_rsi = ta.EMA(rsi, timeperiod=n2)
    return rescale(e_rsi, np.float64(0.), np.float64(100.), np.float64(0.), np.float64(1.))


def n_cci(src: np.ndarray, n1: int, n2: int):
    cci = ta.CCI(src, timeperiod=n1)
    e_cci = ta.EMA(cci, timeperiod=n2)
    return normalize(e_cci, np.float64(0.), np.float64(100.))


def n_wt(src: np.ndarray, n1: int = 10, n2: int = 11):
    ema1 = ta.EMA(src, n1)
    ema2 = ta.EMA(np.abs(src - ema1), n1)
    ci = (src - ema1) / (0.015 * ema2)
    wt1 = ta.EMA(ci, n2)
    wt2 = ta.SMA(wt1, 4)
    normalize(wt1 - wt2, np.float64(0.), np.float64(1.))


def n_adx(adx: np.ndarray, length: int = 14):
    # todo: tr_smooth, smooth_directional_movement_plus, smooth_neg_movement 算法需要内置到 indicator 库中
    return rescale(adx, np.float64(0), np.float64(100), np.float64(0), np.float64(1))


def filter_adx(adx: np.float64, adx_threshold: int, length: int = 14):
    # todo: tr_smooth, smooth_directional_movement_plus, smooth_neg_movement 算法需要内置到 indicator 库中
    return adx > adx_threshold


def regime_filter(normalized_slope_decline: np.float64, threshold: np.float64):
    # todo: regime_val1, regime_val2, regime_klmf 算法需要内置到 indicator 库中
    return normalized_slope_decline >= threshold


def filter_volatility(high: np.ndarray, low: np.ndarray, close: np.ndarray, min_len: int = 1, max_len: int = 10):
    recent_atr = ta.ATR(high, low, close, min_len)
    historical_atr = ta.ATR(high, low, close, max_len)
    return recent_atr > historical_atr
