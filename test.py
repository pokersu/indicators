import time
import pandas as pd
import mplfinance as mpf
import numpy as np
import indicators.indicators as indicators
import indicators.kernel as kernel
import indicators.knn as knn
from indicators.filter import filter_volatility, filter_regime, filter_adx
from datetime import date

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

df = pd.read_feather('./data/DOGE_USDT_USDT-5m-futures.feather')
df.set_index('date', inplace=True)

df = df[df.index.date > date(2023, 8, 21)]

df['n_rsi_141'] = df.n_rsi(14, 1)
df['n_wt'] = df.n_wt(10, 11)
df['n_cci'] = df.n_cci(20, 1)
df['n_adx'] = df.n_adx(20)
df['n_rsi_91'] = df.n_rsi(9, 1)

h, r, x, lag = 8, 8., 25, 2
df['yhat1'] = df.rational_quadratic(h, r, x)
df['yhat2'] = df.gaussian(h - lag, x)

df['train_label'] = df.train_labeling(features=['n_rsi_141', 'n_wt', 'n_cci', 'n_adx', 'n_rsi_91'])

x = df['train_label']
df['filter_all'] = filter_regime(df) & filter_adx(df) & filter_volatility(df)
df['signal'] = df.render_signal()
df['signal_held'] = df.render_signal_held()
df['signal_flip'] = df.render_signal_flip()
df['kernel_trend'] = df.render_kernel_trend()
df['ema_trend'] = df.render_ema_trend(200)
df['sma_trend'] = df.render_sma_trend(200)

# // Bar-Count Filters: Represents strict filters based on a pre-defined holding period of 4 bars
# var int barsHeld = 0
# barsHeld := ta.change(signal) ? 0 : barsHeld + 1
# isHeldFourBars = barsHeld == 4
# isLastSignalBuy = signal[4] == direction.long and isEmaUptrend[4] and isSmaUptrend[4]
# isLastSignalSell = signal[4] == direction.short and isEmaDowntrend[4] and isSmaDowntrend[4]
# isHeldLessThanFourBars = 0 < barsHeld and barsHeld < 4
# isNewBuySignal = isBuySignal and isDifferentSignalType
# isNewSellSignal = isSellSignal and isDifferentSignalType
# endLongTradeStrict = ((isHeldFourBars and isLastSignalBuy) or (isHeldLessThanFourBars and isNewSellSignal and isLastSignalBuy)) and startLongTrade[4]
# endShortTradeStrict = ((isHeldFourBars and isLastSignalSell) or (isHeldLessThanFourBars and isNewBuySignal and isLastSignalSell)) and startShortTrade[4]


df['long'] = df.render_entry()
df['short'] = df.render_exit()

df = df[-1200:]
length = df.shape[0] / 300
buy = np.where(df['long'] > 0, 1, np.nan) * df['low'] - 2
sell = np.where(df['short'] > 0, 1, np.nan) * df['high'] + 2
buy_apd = mpf.make_addplot(buy, scatter=True, markersize=100, marker=r'^', color='green')
sell_apd = mpf.make_addplot(sell, scatter=True, markersize=100, marker=r'v', color='red')
yhat1_apd = mpf.make_addplot(df['yhat1'], color='green')
yhat2_apd = mpf.make_addplot(df['yhat2'], color='red')
mpf.plot(df, type="candle", style="yahoo", volume=True, figsize=(length * 40, 12), addplot=[buy_apd, sell_apd])
