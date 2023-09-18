import time

import pandas as pd
import mplfinance as mpf
import numpy as np
import indicators.indicators as indicators
import indicators.kernel as kernel
import indicators.knn as knn
from datetime import date

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

df = pd.read_feather('./data/ETH_USDT_USDT-5m-futures.feather')
df.set_index('date', inplace=True)

df = df[df.index.date > date(2023, 8, 1)]

df['n_rsi'] = df.n_rsi(14, 1)
df['n_wt'] = df.n_wt(10, 11)
df['n_cci'] = df.n_cci(20, 1)
df['n_adx'] = df.n_adx(20)
df['s_adx'] = df.s_adx(14)
#
# h, r, x, lag = 8, 8., 25, 2
# df['yhat1'] = df.rational_quadratic(h, r, x)
# df['yhat2'] = df.gaussian(h - lag, x)
# df['regime'] = df.regime()
# df['distance'] = df.distance(df.iloc[-1], features=['n_rsi', 'n_wt', 'n_cci', 'n_adx'])

df['trend'] = df.trend()


def filter_volatility(bars: pd.DataFrame, min_length: int = 1, max_length: int = 10):
    recent_atr = bars.atr(min_length)
    historical_atr = bars.atr(max_length)
    return recent_atr > historical_atr


def filter_adx(bars: pd.DataFrame, threshold: int = 20):
    return bars['s_adx'] >= threshold


def filter_regime(bars: pd.DataFrame, threshold: float = -0.1):
    return bars['regime'] >= threshold


features = ['n_rsi', 'n_wt', 'n_cci', 'n_adx']

data_len = len(df)


def kl(row):
    index = row.name
    historic = df[:index].iloc[:-1]
    his_data_len = len(historic)
    if his_data_len < 3000 or data_len - his_data_len > 200:
        return np.nan

    point = row[features].values.astype(np.float64)
    points = historic[features].values.astype(np.float64)
    distance = pd.Series(np.sum(np.log(1 + np.abs(points - point)), axis=1), index=historic.index)

    last_distance = -1.0
    distances = []
    predictions = []
    c = 0
    for index, d in distance.items():
        if d >= last_distance and c % 4 == 0:
            last_distance = d
            distances.insert(0, d)
            predictions.insert(0, historic.loc[index]['trend'])
            if len(distances) >= 8:
                last_distance = distance[5]
                distances.pop()
                predictions.pop()
        c += 1
    return sum(predictions)


t = df.apply(kl, axis=1)

df = df[-600:]
length = df.shape[0] / 300
# buy = np.where((df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)), 1, np.nan) * df['low'] * 0.999
# sell = np.where((df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)), 1, np.nan) * df['high'] * 1.001
# buy = np.where(df['trend'] > 0 & filter_regime(df) & filter_adx(df) & filter_volatility(df), 1, np.nan) * df['low'] - 2
# sell = np.where(df['trend'] < 0, 1, np.nan) * df['high'] + 2
# buy_apd = mpf.make_addplot(buy, scatter=True, markersize=100, marker=r'^', color='green')
# sell_apd = mpf.make_addplot(sell, scatter=True, markersize=100, marker=r'v', color='red')
# yhat1_apd = mpf.make_addplot(df['yhat1'], color='green')
# yhat2_apd = mpf.make_addplot(df['yhat2'], color='red')
# mpf.plot(df, type="candle", style="yahoo", volume=True, figsize=(length * 40, 12), addplot=[buy_apd, sell_apd])

print(df[['close', 'n_rsi', 'n_wt', 'n_cci', 'n_adx', 'yhat1', 'yhat2', 'regime', 'distance', 'trend']].tail(
    50).sort_index(ascending=False))
