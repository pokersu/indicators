import talib
import numpy as np
import numpy.typing as npt
import numpy
import typing


class FeatureArrays:
    def __init__(self):
        self.feature_arrays: typing.List[npt.NDArray[numpy.float64]] = []

    def add_feature_array(self, feature_array: npt.NDArray[numpy.float64]) -> None:
        self.feature_arrays.append(feature_array)

    def cut_data_to_same_lenx(
            self, reference_length: typing.Optional[int] = None
    ) -> int:
        self.feature_arrays = cut_data_to_same_len(
            self.feature_arrays, reference_length=reference_length
        )
        return len(self.feature_arrays[0])

# 这个函数计算相对强弱指数（RSI）。RSI是一种用于衡量资产价格的强度和速度的指标。
# 函数参数_close是一个包含价格数据的数组，f_paramA和f_paramB是函数的参数。
# 函数首先计算RSI，然后对其进行EMA（指数移动平均）处理，最后进行重新缩放。
def n_rsi(_close, f_paramA, f_paramB):
    rsi = talib.RSI(_close, timeperiod=f_paramA)
    ema_rsi = talib.EMA(rsi, timeperiod=f_paramB)
    rescaled_ema_rsi = (ema_rsi - ema_rsi.min()) / (ema_rsi.max() - ema_rsi.min())
    return rescaled_ema_rsi


# 这个函数计算"Wave Trend"指标，通过对价格的EMA和波动性进行计算来识别趋势。
# 函数参数_hlc3是一个包含高、低、收盘价格的数组，f_paramA和f_paramB是函数的参数。
# 函数计算了一系列指标，并进行归一化处理。
def n_wt(_hlc3, f_paramA, f_paramB):
    ema1 = talib.EMA(_hlc3, timeperiod=f_paramA)
    ema2 = talib.EMA(abs(_hlc3 - ema1), timeperiod=f_paramA)
    ci = (_hlc3[1:] - ema1[1:]) / (0.015 * ema2[1:])
    wt1 = talib.EMA(ci, timeperiod=f_paramB)
    wt2 = talib.SMA(wt1, timeperiod=4)
    normalized_wt = (wt1 - wt2 - (wt1 - wt2).min()) / ((wt1 - wt2).max() - (wt1 - wt2).min())
    return normalized_wt


# 这个函数计算"Commodity Channel Index"（CCI）指标，用于衡量价格相对于其统计平均数的偏离程度。
# 函数参数highs、lows和closes是包含高、低、收盘价格的数组，f_paramA和f_paramB是函数的参数。
# 函数计算CCI，然后进行归一化处理
def n_cci(highs, lows, closes, f_paramA, f_paramB):
    cci = talib.CCI(highs, lows, closes, timeperiod=f_paramA)
    normalized_cci = (cci - cci.min()) / (cci.max() - cci.min())
    return normalized_cci


# 这段代码计算"Average Directional Index"（ADX）技术指标。ADX用于衡量趋势的强度，它基于DI+和DI-指标的差异。下面我会解释这段代码，并提供将其优化为使用tALib库的版本。
# 优化建议：
#
# 使用tALib库：将原始的手动计算和数据处理步骤替换为tALib库函数，以提高效率和准确性。
# 避免重复计算：在循环内部进行的一些计算在tALib中已经有内置的函数，可以避免重复计算和错误。
# 数据类型一致性：确保函数参数的数据类型与tALib函数的要求一致，这有助于避免意外错误。
# 归一化处理：在返回之前，对adx进行了归一化处理，使其在0到1之间。这可以帮助确保指标值在统一的范围内。
# 使用NumPy：在进行数组操作时，使用NumPy库可以提高计算效率。将数据类型从tulipy.NDArray更改为numpy.ndarray以与NumPy兼容。
# 注释和文档：为函数和关键计算部分添加适当的注释，以便其他开发人员理解代码的逻辑和目的。
# 异常处理：在实际应用中，可能需要添加适当的异常处理，以处理可能出现的问题，例如数据长度不足等。
# 性能优化：对于大量数据，循环可能会导致性能问题。在处理大量数据时，可以考虑使用更高效的算法或并行计算。
def n_adx(
        highSrc: np.ndarray,
        lowSrc: np.ndarray,
        closeSrc: np.ndarray,
        f_paramA: int,
):
    length = f_paramA
    data_length = len(highSrc)
    trSmooth = [0]
    smoothnegMovement = [0]
    smoothDirectionalMovementPlus = [0]
    dx = []

    for index in range(1, data_length):
        tr = max(
            max(
                highSrc[index] - lowSrc[index],
                abs(highSrc[index] - closeSrc[index - 1]),
            ),
            abs(lowSrc[index] - closeSrc[index - 1]),
        )
        directionalMovementPlus = (
            max(highSrc[index] - highSrc[index - 1], 0)
            if highSrc[index] - highSrc[index - 1] > lowSrc[index - 1] - lowSrc[index]
            else 0
        )
        negMovement = (
            max(lowSrc[index - 1] - lowSrc[index], 0)
            if lowSrc[index - 1] - lowSrc[index] > highSrc[index] - highSrc[index - 1]
            else 0
        )
        trSmooth.append(trSmooth[-1] - trSmooth[-1] / length + tr)
        smoothDirectionalMovementPlus.append(
            smoothDirectionalMovementPlus[-1]
            - smoothDirectionalMovementPlus[-1] / length
            + directionalMovementPlus
        )
        smoothnegMovement.append(
            smoothnegMovement[-1] - smoothnegMovement[-1] / length + negMovement
        )
        diPositive = smoothDirectionalMovementPlus[-1] / trSmooth[-1] * 100
        diNegative = smoothnegMovement[-1] / trSmooth[-1] * 100

        if index > 3:
            # skip early candles as its division by 0
            dx.append(abs(diPositive - diNegative) / (diPositive + diNegative) * 100)
    dx = np.array(dx)
    adx = talib.RMA(dx, timeperiod=length)
    return (adx - adx.min()) / (adx.max() - adx.min())


# 解释：
#
# 这个函数名为series_from，它接受多个参数，包括一个特征字符串和不同的价格和参数数据。
# 函数根据给定的feature_string来判断要计算哪个技术指标，并调用相应的函数来进行计算。
# 如果feature_string是"RSI"，则调用n_rsi函数来计算相对强弱指数。
# 如果feature_string是"WT"，则调用n_wt函数来计算"Wave Trend"指标。
# 如果feature_string是"CCI"，则调用n_cci函数来计算"Commodity Channel Index"指标。
# 如果feature_string是"ADX"，则调用n_adx函数来计算"Average Directional Index"指标。
# 优化建议：
#
# 使用字典映射：你可以考虑使用一个字典来映射特征字符串和相应的函数，这将使代码更具扩展性和可维护性。


def series_from(
        feature_string: str,
        _close: npt.NDArray[numpy.float64],
        _high: npt.NDArray[numpy.float64],
        _low: npt.NDArray[numpy.float64],
        _hlc3: npt.NDArray[numpy.float64],
        f_paramA: int,
        f_paramB: int,
) -> npt.NDArray[numpy.float64]:
    if feature_string == "RSI":
        return n_rsi(_close, f_paramA, f_paramB)
    if feature_string == "WT":
        return n_wt(_hlc3, f_paramA, f_paramB)
    if feature_string == "CCI":
        return n_cci(_high, _low, _close, f_paramA, f_paramB)
    if feature_string == "ADX":
        return n_adx(_high, _low, _close, f_paramA)


# 这个方法名为get_feature_arrays，它接受四个输入参数，分别是蜡烛的收盘价格、最高价、最低价和HLC3（高+低+收盘的均值）价格。
# 方法返回一个FeatureArrays对象，其中包含计算得到的特征数组。
def get_feature_arrays(
        self,
        candle_closes: np.ndarray,
        candle_highs: np.ndarray,
        candle_lows: np.ndarray,
        candles_hlc3: np.ndarray,
) -> FeatureArrays:
    feature_arrays = FeatureArrays()  # 创建一个 FeatureArrays 对象来存储特征数组

    for feature_settings in self.trading_mode.feature_engineering_settings.features_settings:
        # 对于每个特征设置，调用 series_from 函数计算特征数组，并添加到 FeatureArrays 对象中
        feature_array = series_from(
            feature_settings.indicator_name,
            candle_closes,
            candle_highs,
            candle_lows,
            candles_hlc3,
            feature_settings.param_a,
            feature_settings.param_b,
        )
        feature_arrays.add_feature_array(feature_array)

    return feature_arrays
