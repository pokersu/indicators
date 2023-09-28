import pandas as pd


def filter_volatility(bars: pd.DataFrame, min_length: int = 1, max_length: int = 10):
    recent_atr = bars.atr(min_length)
    historical_atr = bars.atr(max_length)
    return recent_atr > historical_atr


def filter_adx(bars: pd.DataFrame, threshold: int = 20):
    s_adx_14 = bars.s_adx(14)
    return s_adx_14 >= threshold


def filter_regime(bars: pd.DataFrame, threshold: float = -0.1):
    regime = bars.regime()
    return regime >= threshold
