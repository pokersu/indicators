import typing
import numpy.typing as npt
import numpy
import enum


class YTrainTypes(enum.Enum):
    # 定义训练数据类型的枚举类

    # 在经过 x 根蜡烛后是否盈利
    IS_IN_PROFIT_AFTER_4_BARS = "is_in_profit_after_x_bars"

    # 基于收盘价，在经过 x 根蜡烛后是否盈利
    IS_IN_PROFIT_AFTER_4_BARS_CLOSES = "is_in_profit_after_x_bars_based_on_closes"

    # 是否是盈利交易
    IS_WINNING_TRADE = "is_winning_trade"


class SignalDirection:
    # 定义信号方向的类

    # 信号方向的整数表示
    long: int = 1  # 多头信号，代表买入信号
    short: int = -1  # 空头信号，代表卖出信号
    neutral: int = 0  # 中性信号，代表无交易信号

    # 信号方向的元组表示
    tuple_long: typing.Tuple[int, int, int] = (1, 0, 0)  # (1, 0, 0) 表示多头信号
    tuple_short: typing.Tuple[int, int, int] = (0, 0, 1)  # (0, 0, 1) 表示空头信号
    tuple_neutral: typing.Tuple[int, int, int] = (0, 1, 0)  # (0, 1, 0) 表示中性信号


def verify_training_prediction_labels_completeness(y_train_series):
    # 验证训练预测标签的完整性
    # 如果短信号、中性信号和长信号都不在 y_train_series 中
    if (
        SignalDirection.short not in y_train_series
        or SignalDirection.neutral not in y_train_series
        or SignalDirection.long not in y_train_series
    ):
        # 抛出运行时错误，提示需要增加可用的历史数据或更改训练预测标签设置
        raise RuntimeError(
            "Not enough historical data available, increase the available history "
            "or change your training prediction labels settings"
        )



def get_y_train_series(
        closes: npt.NDArray[numpy.float64],
        lows: npt.NDArray[numpy.float64],
        highs: npt.NDArray[numpy.float64],
        raise_missing_data: bool = False,
):
    # 获取训练数据的标签（预测目标）
    if training_data_type == YTrainTypes.IS_WINNING_TRADE:
        y_train_series = []
        data_length = len(closes)
        for candle_index, candle in enumerate(closes):
            # 计算不同交易方向的价格
            long_win_price = candle / 100 * (100 + percent_for_a_win)
            long_lose_price = candle / 100 * (100 - percent_for_a_loss)
            short_win_price = candle / 100 * (100 - percent_for_a_win)
            short_lose_price = candle / 100 * (100 + percent_for_a_loss)
            is_short = None
            is_long = None
            signal = SignalDirection.neutral

            # 遍历之后的蜡烛，判断交易方向
            for inner_candle_index in range(1, data_length - candle_index):
                if is_short is False and is_long is False:
                    signal = SignalDirection.neutral
                    break
                comparing_high_candle = highs[candle_index + inner_candle_index]
                comparing_low_candle = lows[candle_index + inner_candle_index]

                # 判断是否可以开多头交易
                if is_long is None:
                    if comparing_high_candle >= long_win_price:
                        signal = SignalDirection.long
                        break
                    if comparing_low_candle <= long_lose_price:
                        is_long = False

                # 判断是否可以开空头交易
                if is_short is None:
                    if comparing_low_candle <= short_win_price:
                        signal = SignalDirection.short
                        break
                    if comparing_high_candle >= short_lose_price:
                        is_short = False
            y_train_series.append(signal)
    elif (
            training_data_type
            == YTrainTypes.IS_IN_PROFIT_AFTER_4_BARS
    ):
        # 使用后4根K线的数据计算标签
        cutted_closes, _ = shift_data(closes, 4)
        _, shifted_lows = shift_data(lows, 4)
        _, shifted_highs = shift_data(highs, 4)
        # 判断是否盈利
        y_train_series = numpy.where(
            shifted_highs < cutted_closes,
            SignalDirection.short,
            numpy.where(
                shifted_lows > cutted_closes,
                SignalDirection.long,
                SignalDirection.neutral,
            ),
        )
    elif (
            training_data_type
            == YTrainTypes.IS_IN_PROFIT_AFTER_4_BARS_CLOSES
    ):
        # 使用后4根K线的收盘价数据计算标签
        cutted_closes, shifted_closes = shift_data(closes, 4)
        # 判断是否盈利
        y_train_series = numpy.where(
            shifted_closes < cutted_closes,
            SignalDirection.short,
            numpy.where(
                shifted_closes > cutted_closes,
                SignalDirection.long,
                SignalDirection.neutral,
            ),
        )

    # 如果需要，检查是否存在缺失数据
    if raise_missing_data:
        verify_training_prediction_labels_completeness(y_train_series)

    # 返回训练数据的标签
    return y_train_series


def get_candles_back_start_end_index(current_candle_index: int):
    # 计算循环的大小，最大为 classification_settings_max_bars_back - 1 或当前蜡烛索引
    size_loop: int = min(
        classification_settings_max_bars_back - 1,
        current_candle_index,
    )

    # 根据是否使用远程分形来确定起始和结束索引
    if classification_settings_use_remote_fractals:
        # 使用远程分形时的计算逻辑
        # 对于实时模式：从第一根蜡烛开始分类
        # 对于回测模式：从当前蜡烛索引减去 live_history_size 开始分类
        start_index: int = max(
            current_candle_index - classification_settings_live_history_size,
            0,
        )
        end_index: int = start_index + size_loop
    else:
        # 不使用远程分形时的计算逻辑
        # 从当前蜡烛索引减去 size_loop 开始分类，结束于当前蜡烛索引
        start_index: int = current_candle_index - size_loop
        end_index: int = current_candle_index

    # 返回一个范围对象，表示起始和结束索引之间的范围
    return range(start_index, end_index)


def get_lorentzian_distance(
        candle_index: int,
        candles_back_index: int,
        feature_arrays: FeatureArrays,
) -> float:
    # 初始化距离为0
    distance: float = 0

    # 遍历特征数组，计算洛伦兹距离
    for feature_array in feature_arrays.feature_arrays:
        # 计算当前特征在两个蜡烛之间的差异，取绝对值后加1再取对数，然后累加到距离
        distance += math.log(
            1 + abs(feature_array[candle_index] - feature_array[candles_back_index])
        )

    # 返回计算得到的距离
    return distance


def get_classification_predictions(
        current_candle_index: int, feature_arrays, y_train_series
) -> int:
    # 初始化变量
    last_distance: float = -1  # 上一个距离，默认为-1
    predictions: list = []  # 存储预测值的列表
    distances: list = []  # 存储距离值的列表

    # 遍历历史蜡烛数据，计算预测值
    for candles_back in get_candles_back_start_end_index(current_candle_index):
        # 判断是否满足下采样条件
        if classification_settings_down_sampler(
                candles_back,
                only_train_on_every_x_bars,
        ):
            # 计算洛伦兹距离
            lorentzian_distance: float = get_lorentzian_distance(
                candle_index=current_candle_index,
                candles_back_index=candles_back,
                feature_arrays=feature_arrays,
            )
            # 如果洛伦兹距离大于等于上一个距离
            if lorentzian_distance >= last_distance:
                last_distance = lorentzian_distance
                predictions.append(y_train_series[candles_back])
                distances.append(lorentzian_distance)

                # 控制预测值和距离值的列表长度
                if len(predictions) > classification_settings_neighbors_count:
                    last_distance = distances[
                        classification_settings_last_distance_neighbors_count
                    ]
                    del distances[0]
                    del predictions[0]

    # 返回预测值的总和
    return sum(predictions)


def _handle_four_bar_exit(
        bars_since_green_entry: int,
        bars_since_red_entry: int,
        exit_short_trades: list,
        exit_long_trades: list,
        start_long_trade: bool,
        start_short_trade: bool,
) -> typing.Tuple[int, int]:
    # 处理基于四根K线的退出策略
    # 参数解释：
    # bars_since_green_entry: 入场后绿色K线的数量
    # bars_since_red_entry: 入场后红色K线的数量
    # exit_short_trades: 退出短交易的列表
    # exit_long_trades: 退出多头交易的列表
    # start_long_trade: 开启多头交易的标志
    # start_short_trade: 开启空头交易的标志

    # 根据情况更新入场后的K线数
    if start_long_trade:
        bars_since_green_entry = 0
    else:
        bars_since_green_entry += 1
    if start_short_trade:
        bars_since_red_entry = 0
    else:
        bars_since_red_entry += 1

    # 判断是否需要退出交易
    if bars_since_red_entry == 4:
        exit_short_trades.append(True)
        exit_long_trades.append(False)
    elif bars_since_green_entry == 4:
        exit_long_trades.append(True)
        exit_short_trades.append(False)
    else:
        # 根据绿色入场K线数和红色入场K线数判断是否退出
        if bars_since_red_entry < 4 and start_long_trade:
            exit_short_trades.append(True)
        else:
            exit_short_trades.append(False)
        if bars_since_green_entry < 4 and start_short_trade:
            exit_long_trades.append(True)
        else:
            exit_long_trades.append(False)

    # 返回更新后的入场K线数
    return bars_since_green_entry, bars_since_red_entry


def set_signals_from_prediction(
        prediction: int,
        _filters: Filter,
        candle_index: int,
        previous_signals: list,
        start_long_trades: list,
        start_short_trades: list,
        is_bullishs: npt.NDArray[numpy.bool_],
        is_bearishs: npt.NDArray[numpy.bool_],
        exit_short_trades: list,
        exit_long_trades: list,
        bars_since_green_entry: int,
        bars_since_red_entry: int,
        is_buy_signals: list,
        is_sell_signals: list,
        exit_type: str,
) -> typing.Tuple[int, int]:
    # 基于预测和过滤器生成信号
    signal = (
        SignalDirection.long
        if prediction > classification_settings_required_neighbors
           and _filters.filter_all[candle_index]
        else (
            SignalDirection.short
            if prediction < -classification_settings_required_neighbors
               and _filters.filter_all[candle_index]
            else previous_signals[-1]
        )
    )

    # 检查信号类型是否改变
    is_different_signal_type: bool = previous_signals[-1] != signal
    previous_signals.append(signal)

    # 根据预测和趋势确定买入和卖出信号
    is_buy_signal = (
            signal == SignalDirection.long and _filters.is_uptrend[candle_index]
    )
    is_buy_signals.append(is_buy_signal)
    is_sell_signal = (
            signal == SignalDirection.short and _filters.is_downtrend[candle_index]
    )
    is_sell_signals.append(is_sell_signal)

    # 检查是否有新的买入和卖出信号
    is_new_buy_signal = is_buy_signal and is_different_signal_type
    is_new_sell_signal = is_sell_signal and is_different_signal_type

    # 根据条件判断是否适合开启多头交易
    start_long_trade = (
            is_new_buy_signal
            and is_bullishs[candle_index]
            and _filters.is_uptrend[candle_index]
    )
    start_long_trades.append(start_long_trade)

    # 根据条件判断是否适合开启空头交易
    start_short_trade = (
            is_new_sell_signal
            and is_bearishs[candle_index]
            and _filters.is_downtrend[candle_index]
    )
    start_short_trades.append(start_short_trade)

    # 根据退出类型处理退出策略
    if exit_type == ExitTypes.FOUR_BARS:
        # 通过4根K线计数的筛选器：基于预定义的4根K线持有期的严格筛选器
        bars_since_green_entry, bars_since_red_entry = _handle_four_bar_exit(
            bars_since_green_entry,
            bars_since_red_entry,
            exit_short_trades,
            exit_long_trades,
            start_long_trade,
            start_short_trade,
        )
    return bars_since_green_entry, bars_since_red_entry


def classify_current_candle(
    y_train_series: npt.NDArray[numpy.float64],
    current_candle_index: int,
    feature_arrays: FeatureArrays,
    historical_predictions: list,
    _filters: Filter,
    previous_signals: list,
    is_bullishs: npt.NDArray[numpy.bool_],
    is_bearishs: npt.NDArray[numpy.bool_],
    # alerts_bullish: npt.NDArray[numpy.bool_],
    # alerts_bearish: npt.NDArray[numpy.bool_],
    # is_bearish_changes: npt.NDArray[numpy.bool_],
    # is_bullish_changes: npt.NDArray[numpy.bool_],
    bars_since_red_entry: int,
    bars_since_green_entry: int,
    start_long_trades: list,
    start_short_trades: list,
    exit_short_trades: list,
    exit_long_trades: list,
    is_buy_signals: list,
    is_sell_signals: list,
) -> typing.Tuple[int, int]:
    # 计算当前蜡烛的分类预测并添加到历史预测列表中
    historical_predictions.append(
        get_classification_predictions(
            current_candle_index,
            feature_arrays,
            y_train_series,
        )
    )
    # 设置信号和处理退出策略
    (
        bars_since_green_entry,
        bars_since_red_entry,
    ) = set_signals_from_prediction(
        prediction=historical_predictions[-1],
        _filters=_filters,
        candle_index=current_candle_index,
        previous_signals=previous_signals,
        start_long_trades=start_long_trades,
        start_short_trades=start_short_trades,
        is_bullishs=is_bullishs,
        is_bearishs=is_bearishs,
        # alerts_bullish=alerts_bullish,
        # alerts_bearish=alerts_bearish,
        # is_bearish_changes=is_bearish_changes,
        # is_bullish_changes=is_bullish_changes,
        exit_short_trades=exit_short_trades,
        exit_long_trades=exit_long_trades,
        bars_since_green_entry=bars_since_green_entry,
        bars_since_red_entry=bars_since_red_entry,
        is_buy_signals=is_buy_signals,
        is_sell_signals=is_sell_signals,
        exit_type=lorentzian_order_exit_type,
    )
    return bars_since_green_entry, bars_since_red_entry

