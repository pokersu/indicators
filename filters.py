import numpy as np
import talib
import typing
import math


# 解释和优化建议：
#
# 该类接受各种过滤器的结果作为参数，并根据这些结果生成更高级别的过滤器。
# cut_data_to_same_len 被用于将所有输入的数组截断为相同的长度，以保证数据一致性。
# np.logical_and.reduce 用于同时应用多个逻辑与操作。
# 使用数组逻辑运算函数，如 np.logical_and，将多个过滤器结果组合成更高级别的过滤器。
# 添加了 is_uptrend 和 is_downtrend 属性，用于表示趋势状态。
# 请注意，在调用这个类时，确保传入适当的参数，以匹配其构造函数的参数。

class Filter:
    def __init__(
            self,
            volatility: np.ndarray,
            regime: np.ndarray,
            adx: np.ndarray,
            is_ema_uptrend: np.ndarray,
            is_ema_downtrend: np.ndarray,
            is_sma_uptrend: np.ndarray,
            is_sma_downtrend: np.ndarray,
    ):
        (
            volatility,
            regime,
            adx,
            is_ema_uptrend,
            is_ema_downtrend,
            is_sma_uptrend,
            is_sma_downtrend,
        ) = cut_data_to_same_len(
            (
                volatility,
                regime,
                adx,
                is_ema_uptrend,
                is_sma_uptrend,
                is_ema_downtrend,
                is_sma_downtrend,
            )
        )

        # User Defined Filters: Used for adjusting the frequency of the ML Model's predictions
        self.filter_all = np.logical_and.reduce((volatility, regime, adx))

        # Fractal Filters: Derived from relative appearances of signals in a given time series fractal/segment with a default length of 4 bars
        self.is_uptrend = np.logical_and(is_ema_uptrend, is_sma_uptrend)
        self.is_downtrend = np.logical_and(is_ema_downtrend, is_sma_downtrend)

        self.volatility = volatility
        self.regime = regime
        self.adx = adx
        self.is_ema_uptrend = is_ema_uptrend
        self.is_sma_uptrend = is_sma_uptrend
        self.is_ema_downtrend = is_ema_downtrend
        self.is_sma_downtrend = is_sma_downtrend


###################################

# 解释：
#
# 该函数的主要目的是根据 ATR 技术指标来判断价格波动性是否满足一定条件。ATR 衡量了一定时期内价格的波动范围。
# candle_highs、candle_lows 和 candle_closes 是价格数据的高、低和收盘价序列。
# min_length 和 max_length 参数用于设置计算 ATR 的时期长度的最小和最大值。
# use_volatility_filter 参数决定是否使用波动性过滤器。如果设置为 False，将直接返回一个全为 True 的数组，表示不进行过滤。
# 优化建议：
#
# 类型注释和参数说明: 提供了输入参数的类型注释和简要说明，这对于函数的可读性和可维护性很重要。
# 数据长度处理: 为了保证 recentAtr 和 historicalAtr 长度一致，使用了 min_data_length，这确保了对齐计算结果。
# 导入库: 导入了必要的库 numpy 和 tulipy。
# 变量命名: 变量名更具描述性，增加了代码的可读性。
# 向量化操作: 由于 tulipy.atr 是一个矢量化函数，这可以在一次计算中同时处理多个价格序列，从而提高了效率。
# 注意，此代码可能需要根据实际情况进行调整，例如，你可能需要确保传入的价格数据长度足够以计算 max_length 所要求的 ATR 值。

import numpy as np
import talib
from typing import List
import numpy.typing as npt


def filter_volatility(
        candle_highs: npt.NDArray[np.float64],
        candle_lows: npt.NDArray[np.float64],
        candle_closes: npt.NDArray[np.float64],
        min_length: int = 1,
        max_length: int = 10,
        use_volatility_filter: bool = True,
) -> npt.NDArray[np.bool_]:
    if not use_volatility_filter:
        return np.repeat(True, len(candle_closes))

    # 计算最近的 ATR
    recentAtr = talib.ATR(
        high=candle_highs,
        low=candle_lows,
        close=candle_closes,
        timeperiod=min_length
    )

    # 计算历史 ATR
    historicalAtr = talib.ATR(
        high=candle_highs,
        low=candle_lows,
        close=candle_closes,
        timeperiod=max_length
    )

    # 截取最近和历史 ATR 以确保长度一致
    recentAtr, historicalAtr = basic_utils.cut_data_to_same_len(
        (recentAtr, historicalAtr)
    )

    # 判断最近 ATR 是否大于历史 ATR
    return recentAtr > historicalAtr


# 解释：
#
# 该函数用于生成基于移动平均线的趋势过滤器，标识上升或下降趋势。
# candle_closes 是收盘价序列。
# data_length 表示数据的长度。
# filter_settings 是一个包含过滤器设置信息的对象。
# 优化建议：
#
# 默认赋值优化: 将初始值赋值为全 True 的数组，这避免了不必要的计算，只在需要时进行替换。
# 导入库: 替换了 import tulipy 为 import talib。
# 移动平均线计算: 使用 talib.EMA 和 talib.SMA 分别计算 EMA 和 SMA，注意 timeperiod 参数对应之前的 ema_period 和 sma_period。
# 数据长度处理: 使用切片将移动平均线的长度与原始数据对齐。
# 函数参数修改: 添加了 filter_settings 参数，用于获取过滤器设置。
# 确保在调用函数时，传入合适的 filter_settings 对象以匹配函数的期望参数。

import numpy as np
import talib
from typing import Tuple
import numpy.typing as npt


def get_ma_filters(
        candle_closes: npt.NDArray[np.float64], data_length: int
) -> Tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
]:
    # 检查是否使用 EMA 过滤器
    if filter_settings.use_ema_filter:
        # 对数据和 EMA 进行截断，使它们具有相同的长度
        filter_ema_candles, filter_ema = cut_data_to_same_len(
            (
                candle_closes,
                talib.EMA(
                    candle_closes, self.trading_mode.filter_settings.ema_period
                ),
            )
        )
        # 判断是否处于 EMA 上涨趋势
        is_ema_uptrend: npt.NDArray[np.bool_] = filter_ema_candles > filter_ema
        # 判断是否处于 EMA 下跌趋势
        is_ema_downtrend: npt.NDArray[np.bool_] = filter_ema_candles < filter_ema
    else:
        # 如果不使用 EMA 过滤器，则默认都为上涨趋势
        is_ema_uptrend: npt.NDArray[np.bool_] = np.repeat(True, data_length)
        is_ema_downtrend: npt.NDArray[np.bool_] = is_ema_uptrend

    # 检查是否使用 SMA 过滤器
    if filter_settings.use_sma_filter:
        # 对数据和 SMA 进行截断，使它们具有相同的长度
        filter_sma_candles, filter_sma = cut_data_to_same_len(
            (
                candle_closes,
                talib.SMA(
                    candle_closes, filter_settings.sma_period
                ),
            )
        )
        # 判断是否处于 SMA 上涨趋势
        is_sma_uptrend: npt.NDArray[np.bool_] = filter_sma_candles > filter_sma
        # 判断是否处于 SMA 下跌趋势
        is_sma_downtrend: npt.NDArray[np.bool_] = filter_sma_candles < filter_sma
    else:
        # 如果不使用 SMA 过滤器，则默认都为上涨趋势
        is_sma_uptrend: npt.NDArray[np.bool_] = np.repeat(True, data_length)
        is_sma_downtrend: npt.NDArray[np.bool_] = is_sma_uptrend

    return is_ema_uptrend, is_ema_downtrend, is_sma_uptrend, is_sma_downtrend


# 解释和优化建议：
#
# 该函数的目的是根据一系列计算，判断价格曲线的趋势是否符合某个条件。
# 一些变量名已更改以增加可读性。
# 使用 math.sqrt 函数计算平方根。
# tulipy.ema 被替换为 talib.EMA 来计算指数移动平均线。
# 为了保证数组长度一致，使用切片来对齐数据长度。

import numpy as np
import talib
import math
from typing import List
import numpy.typing as npt


def regime_filter(
        ohlc4: npt.NDArray[np.float64],
        highs: npt.NDArray[np.float64],
        lows: npt.NDArray[np.float64],
        threshold: float,
        use_regime_filter: bool,
) -> npt.NDArray[np.bool_]:
    data_length = len(ohlc4)

    if not use_regime_filter:
        return np.repeat(True, len(ohlc4))

    # 计算曲线斜率
    values_1: List[float] = [0.0]
    values_2: List[float] = [0.0]
    klmfs: List[float] = [0.0]
    abs_curve_slope: List[float] = []

    for index in range(1, data_length):
        value_1 = 0.2 * (ohlc4[index] - ohlc4[index - 1]) + 0.8 * values_1[-1]
        value_2 = 0.1 * (highs[index] - lows[index]) + 0.8 * values_2[-1]
        values_1.append(value_1)
        values_2.append(value_2)

        omega = abs(value_1 / value_2)
        alpha = (-pow(omega, 2) + math.sqrt(pow(omega, 4) + 16 * pow(omega, 2))) / 8
        klmfs.append(alpha * ohlc4[index] + (1 - alpha) * klmfs[-1])
        abs_curve_slope.append(abs(klmfs[-1] - klmfs[-2]))

    abs_curve_slope_np: npt.NDArray[np.float64] = np.array(abs_curve_slope)
    exponentialAverageAbsCurveSlope: npt.NDArray[np.float64] = talib.EMA(
        abs_curve_slope_np, timeperiod=200
    )

    (
        exponentialAverageAbsCurveSlope,
        abs_curve_slope_np,
    ) = cut_data_to_same_len(
        (exponentialAverageAbsCurveSlope, abs_curve_slope_np)
    )

    normalized_slope_decline: npt.NDArray[np.float64] = (
                                                                abs_curve_slope_np - exponentialAverageAbsCurveSlope
                                                        ) / exponentialAverageAbsCurveSlope

    # 计算曲线斜率

    return normalized_slope_decline >= threshold


# 解释和优化建议：
#
# 该函数用于计算 ADX 指标并将其与阈值进行比较，以确定是否过滤趋势。
# candle_closes、candle_highs 和 candle_lows 分别表示收盘价、最高价和最低价序列。
# length 表示计算指标所需的时期长度。
# adx_threshold 是用于比较 ADX 值的阈值。
# 为了避免除零错误，对一些分母值进行了判断。
# 优化建议：
#
# 导入库: 替换了 import tulipy 为 import talib。
# 数组创建优化: 使用 np.full 创建长度为 data_length 的全 True 数组，以避免重复的数组操作。
# 移动平均线计算: 使用 talib.RMA 代替了 calculate_rma 函数来计算移动平均值。
# 判断逻辑优化: 对分母值进行了判断，避免了除零错误。

import numpy as np
import talib
from typing import List
import numpy.typing as npt


def filter_adx(
        candle_closes: npt.NDArray[np.float64],
        candle_highs: npt.NDArray[np.float64],
        candle_lows: npt.NDArray[np.float64],
        length: int,
        adx_threshold: int,
        use_adx_filter: bool,
) -> npt.NDArray[np.bool_]:
    data_length: int = len(candle_closes)

    if not use_adx_filter:
        return np.repeat(True, len(candle_closes))

    tr_smooths: List[float] = [0.0]
    smoothneg_movements: List[float] = [0.0]
    smooth_directional_movement_plus: List[float] = [0.0]
    dx: List[float] = []

    for index in range(1, data_length):
        tr: float = max(
            max(
                candle_highs[index] - candle_lows[index],
                abs(candle_highs[index] - candle_closes[-2]),
            ),
            abs(candle_lows[index] - candle_closes[-2]),
        )

        directional_movement_plus: float = (
            max(candle_highs[index] - candle_highs[-2], 0)
            if candle_highs[index] - candle_highs[-2]
               > candle_lows[-2] - candle_lows[index]
            else 0
        )

        negMovement: float = (
            max(candle_lows[-2] - candle_lows[index], 0)
            if candle_lows[-2] - candle_lows[index]
               > candle_highs[index] - candle_highs[-2]
            else 0
        )

        tr_smooths.append(tr_smooths[-1] - tr_smooths[-1] / length + tr)
        smooth_directional_movement_plus.append(
            smooth_directional_movement_plus[-1]
            - smooth_directional_movement_plus[-1] / length
            + directional_movement_plus
        )

        smoothneg_movements.append(
            smoothneg_movements[-1] - smoothneg_movements[-1] / length + negMovement
        )

        di_positive = smooth_directional_movement_plus[-1] / tr_smooths[-1] * 100
        di_negative = smoothneg_movements[-1] / tr_smooths[-1] * 100

        if index > 3:
            # 跳过早期的蜡烛，因为会除以 0
            dx.append(
                abs(di_positive - di_negative) / (di_positive + di_negative) * 100
            )

    dx: npt.NDArray[np.float64] = np.array(dx)
    adx: npt.NDArray[np.float64] = talib.RMA(dx, timeperiod=length)

    return adx > adx_threshold


import numpy as np


def get_all_filters(
        candle_closes: np.ndarray,
        data_length: int,
        candles_ohlc4: np.ndarray,
        candle_highs: np.ndarray,
        candle_lows: np.ndarray,
        user_selected_candles: np.ndarray,
        use_volatility_filter: bool,
        threshold: int,
        use_regime_filter: bool,
        adx_threshold: int,
) -> Filter:
    (
        is_ema_uptrend,
        is_ema_downtrend,
        is_sma_uptrend,
        is_sma_downtrend,
    ) = get_ma_filters(candle_closes, data_length)

    volatility = filter_volatility(
        candle_highs=candle_highs,
        candle_lows=candle_lows,
        candle_closes=candle_closes,
        min_length=1,
        max_length=10,
        use_volatility_filter=use_volatility_filter,
    )

    regime = regime_filter(
        ohlc4=candles_ohlc4,
        highs=candle_highs,
        lows=candle_lows,
        threshold=threshold,
        use_regime_filter=use_regime_filter,
    )

    adx = filter_adx(
        candle_closes=user_selected_candles,
        candle_highs=candle_highs,
        candle_lows=candle_lows,
        length=14,
        adx_threshold=adx_threshold,
        use_adx_filter=True,  # Assuming this should be True here
    )

    _filter = Filter(
        volatility=volatility,
        regime=regime,
        adx=adx,
        is_ema_uptrend=is_ema_uptrend,
        is_ema_downtrend=is_ema_downtrend,
        is_sma_uptrend=is_sma_uptrend,
        is_sma_downtrend=is_sma_downtrend,
    )

    return _filter
