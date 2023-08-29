def evaluate_lorentzian_classification():
    # 评估 Lorentzian 分类

    # 获取蜡烛数据
    data_source_symbol: str = "BTC/USDT"
    (
        candle_closes,
        candle_highs,
        candle_lows,
        candles_hlc3,
        candles_ohlc4,
        user_selected_candles,
        candle_times,
    ) = get_candle_data(candle_source_name=candle_source_name,
                        data_source_symbol=data_source_symbol, )
    data_length: int = len(candle_highs)

    # 获取筛选器
    _filters: Filter = get_all_filters(
        candle_closes,
        data_length,
        candles_ohlc4,
        candle_highs,
        candle_lows,
        user_selected_candles,
    )

    # 获取内核数据
    (
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
    ) = get_kernel_data(user_selected_candles, data_length)

    # 获取特征数组
    feature_arrays: FeatureArrays = get_feature_arrays(
        candle_closes=candle_closes,
        candle_highs=candle_highs,
        candle_lows=candle_lows,
        candles_hlc3=candles_hlc3,
    )

    # 获取训练数据的标签
    y_train_series: npt.NDArray[
        numpy.bool_
    ] = get_y_train_series(
        candle_closes,
        candle_highs,
        candle_lows,
    )

    # 对所有历史数据进行截断，使长度相同
    (
        y_train_series,
        _filters.filter_all,
        _filters.is_uptrend,
        _filters.is_downtrend,
        candle_closes,
        candle_highs,
        candle_lows,
        candle_times,
        candles_hlc3,
        user_selected_candles,
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
    ) = cut_data_to_same_len(
        (
            y_train_series,
            _filters.filter_all,
            _filters.is_uptrend,
            _filters.is_downtrend,
            candle_closes,
            candle_highs,
            candle_lows,
            candle_times,
            candles_hlc3,
            user_selected_candles,
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
        ),
        reference_length=feature_arrays.cut_data_to_same_lenx(),
    )

    # 获取截断后的数据长度
    cutted_data_length: int = feature_arrays.cut_data_to_same_lenx(
        reference_length=len(candle_closes)
    )

    # 根据情况获取最大历史数据索引
    if (
            not is_backtesting
            and is_plot_recording_mode
    ):
        max_bars_back_index: int = (
            cutted_data_length - 200 if cutted_data_length > 200 else 0
        )
    else:
        max_bars_back_index: int = get_max_bars_back_index(cutted_data_length)

    # 初始化变量
    previous_signals: list = [SignalDirection.neutral]
    historical_predictions: list = []
    bars_since_red_entry: int = 5  # 不在循环开始时触发退出
    bars_since_green_entry: int = 5  # 不在循环开始时触发退出

    start_long_trades: list = []
    start_short_trades: list = []
    exit_short_trades: list = []
    exit_long_trades: list = []
    is_buy_signals: list = []
    is_sell_signals: list = []

    # 遍历历史数据
    for candle_index in range(max_bars_back_index, cutted_data_length):
        (
            bars_since_green_entry,
            bars_since_red_entry,
        ) = classify_current_candle(
            y_train_series=y_train_series,
            current_candle_index=candle_index,
            feature_arrays=feature_arrays,
            historical_predictions=historical_predictions,
            _filters=_filters,
            previous_signals=previous_signals,
            is_bullishs=is_bullishs,
            is_bearishs=is_bearishs,
            # alerts_bullish=alerts_bullish,
            # alerts_bearish=alerts_bearish,
            # is_bearish_changes=is_bearish_changes,
            # is_bullish_changes=is_bullish_changes,
            bars_since_red_entry=bars_since_red_entry,
            bars_since_green_entry=bars_since_green_entry,
            start_long_trades=start_long_trades,
            start_short_trades=start_short_trades,
            exit_short_trades=exit_short_trades,
            exit_long_trades=exit_long_trades,
            is_buy_signals=is_buy_signals,
            is_sell_signals=is_sell_signals,
        )
