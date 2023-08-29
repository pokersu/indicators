import numpy as np
import math
import typing


# 解释和优化建议：
#
# 这个函数实现了一个有理二次滤波器，用于平滑数据序列。
# data_source 是输入的数据序列，look_back 是回溯窗口大小，relative_weight 是相对权重参数，start_at_Bar 是开始计算的位置。
# 首先，将 start_at_Bar 增加 1，这可能是为了与某些其他工具的计算方式保持一致。
# 然后，函数从 start_at_Bar 开始遍历数据，对每个点应用有理二次滤波器计算平滑值。
# 代码中使用了 pow 函数来进行幂运算，np.array 用于将结果转换为数组。
# 优化建议：
#
# 代码看起来已经相当简洁，但可以考虑使用向量化操作来提高效率。例如，可以使用 np.arange 生成一组索引，然后计算权重，最后使用数组运算计算加权平均。
# 另外，确保在调用函数时，传入适当的参数，以获得期望的结果。
import numpy as np
import talib
from typing import List
import numpy.typing as npt

def rationalQuadratic(
    data_source: npt.NDArray[np.float64],
    look_back: int,
    relative_weight: float,
    start_at_Bar: int,
) -> npt.NDArray[np.float64]:
    yhat: List[float] = []
    start_at_Bar += 1  # 因为这在TradingView中是1，而不是0
    for index in range(start_at_Bar, len(data_source)):
        _currentWeight: float = 0
        _cumulativeWeight: float = 0
        for bars_back_index in range(0, start_at_Bar):
            y = data_source[index - bars_back_index]
            w = pow(
                1
                + (
                    pow(bars_back_index, 2)
                    / ((pow(look_back, 2) * 2 * relative_weight))
                ),
                -relative_weight,
            )
            _currentWeight += y * w
            _cumulativeWeight += w
        yhat.append(_currentWeight / _cumulativeWeight)
    return np.array(yhat)



# 解释和优化建议：
#
# 这个函数实现了一个高斯滤波器，用于平滑数据序列。
# data_source 是输入的数据序列，look_back 是回溯窗口大小，start_at_Bar 是开始计算的位置。
# 首先，将 start_at_Bar 增加 1，可能是为了与某些其他工具的计算方式保持一致。
# 然后，函数从 start_at_Bar 开始遍历数据，对每个点应用高斯滤波器计算平滑值。
# 代码中使用了 math.exp 函数来进行指数运算，np.array 用于将结果转换为数组。
# 优化建议：
#
# 类似于之前提到的有理二次滤波器，可以考虑使用向量化操作来提高效率。例如，可以使用 np.arange 生成一组索引，然后计算权重，最后使用数组运算计算加权平均。
# 另外，确保在调用函数时，传入适当的参数，以获得期望的结果。
import numpy as np
import talib
import math
from typing import List
import numpy.typing as npt


def gaussian(
        data_source: npt.NDArray[np.float64], look_back: int, start_at_Bar: int
) -> npt.NDArray[np.float64]:
    start_at_Bar += 1
    yhat: List[float] = []

    for index in range(start_at_Bar, len(data_source)):
        _currentWeight: float = 0
        _cumulativeWeight: float = 0

        for bars_back_index in range(0, start_at_Bar):
            y = data_source[index - bars_back_index]
            w = math.exp(-pow(bars_back_index, 2) / (2 * pow(look_back, 2)))
            _currentWeight += y * w
            _cumulativeWeight += w

        yhat.append(_currentWeight / _cumulativeWeight)

    return np.array(yhat)


#
# 解释和优化建议：
#
# 这个函数接受两个数据序列 data1 和 data2，然后检测它们的交叉情况，分为向上交叉和向下交叉。
# shift_data 函数可能是用于将数据序列向后平移一个位置，以便在检测交叉时使用。
# 使用 np.logical_and 函数来进行逻辑与操作，从而检测交叉的条件。
# 返回一个元组，包含向上交叉和向下交叉的布尔数组。
# 优化建议：
#
# 代码已经相当简洁，没有明显的优化空间。
# 如果有可能，可以确保在调用这个函数时传入合适的数据序列，以获得预期的结果。

import numpy as np
import talib
from typing import Tuple
import numpy.typing as npt


def get_is_crossing_data(
        data1: npt.NDArray[np.float64], data2: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    data1_cutted_1, data1_shifted_1 = basic_utils.shift_data(data1, 1)
    data2_cutted_1, data2_shifted_1 = basic_utils.shift_data(data2, 1)

    # 判断交叉上穿和交叉下穿
    crossing_ups = np.logical_and(
        data1_shifted_1 < data2_shifted_1, data1_cutted_1 > data2_cutted_1
    )
    crossing_downs = np.logical_and(
        data1_shifted_1 > data2_shifted_1, data1_cutted_1 < data2_cutted_1
    )

    return crossing_ups, crossing_downs


# 解释：
#
# 这个函数通过调用 rationalQuadratic 和 gaussian 函数计算了两种不同类型的核估计。
# 根据核估计计算了 Kernel Rates of Change。
# 根据两种核估计计算了交叉和平滑信号。
# 基于信号生成了牛市和熊市的警报。
# 基于信号生成了牛市和熊市的过滤器。
# 返回了一系列信号、警报和估计。
def get_kernel_data(user_selected_candles: np.ndarray, data_length: int) -> typing.Tuple:
    yhat1 = rationalQuadratic(
        user_selected_candles,
        kernel_lookback_window,
        kernel_relative_weighting,
        kernel_regression_level,
    )
    yhat2 = gaussian(
        user_selected_candles,
        kernel_lookback_window - kernel_lag,
        kernel_regression_level,
    )
    yhat1, yhat2 = cut_data_to_same_len((yhat1, yhat2))

    kernel_estimate = yhat1

    yhat1_cutted_1, yhat1_shifted_1 = shift_data(yhat1, 1)
    yhat1_cutted_2, yhat1_shifted_2 = shift_data(yhat1, 2)
    was_bearish_rates = yhat1_shifted_2 > yhat1_cutted_2
    was_bullish_rates = yhat1_shifted_2 < yhat1_cutted_2

    is_bearish_rates = yhat1_shifted_1 > yhat1_cutted_1
    is_bullish_rates = yhat1_shifted_1 < yhat1_cutted_1

    is_bearish_rates, was_bullish_rates = cut_data_to_same_len(
        (is_bearish_rates, was_bullish_rates)
    )
    is_bearish_changes = np.logical_and(
        is_bearish_rates, was_bullish_rates
    )
    is_bullish_rates, was_bearish_rates = cut_data_to_same_len(
        (is_bullish_rates, was_bearish_rates)
    )
    is_bullish_changes = np.logical_and(
        is_bullish_rates, was_bearish_rates
    )

    is_bullish_cross_alerts, is_bearish_cross_alerts = get_is_crossing_data(
        yhat2, yhat1
    )
    is_bullish_smooths = yhat2 >= yhat1
    is_bearish_smooths = yhat2 <= yhat1

    alerts_bullish = is_bullish_cross_alerts if use_kernel_smoothing else is_bullish_changes
    alerts_bearish = is_bearish_cross_alerts if use_kernel_smoothing else is_bearish_changes

    is_bullishs = (
        is_bullish_smooths
        if use_kernel_smoothing
        else is_bullish_rates
    ) if use_kernel_filter else np.repeat(True, data_length)

    is_bearishs = (
        is_bearish_smooths
        if use_kernel_smoothing
        else is_bearish_rates
    ) if use_kernel_filter else np.repeat(True, data_length)

    return (
        alerts_bullish,
        alerts_bearish,
        is_bullishs,
        is_bearishs,
        is_bearish_changes,
        is_bullish_changes,
        is_bullish_cross_alerts,
        is_bearish_cross_alerts,
        kernel_estimate,
        yhat2,
        is_bearish_rates,
        was_bullish_rates,
        is_bullish_rates,
        was_bearish_rates,
    )
