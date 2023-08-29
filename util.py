async def get_candle_data(
        candle_source_name: str,
        data_source_symbol: str,
) -> tuple:
    # 获取蜡烛数据

    # 根据是否在回测模式中决定是否获取完整历史数据
    max_history = True if is_backtesting else False

    # 获取蜡烛时间
    candle_times = await exchange_public_data.Time(
        symbol=data_source_symbol, max_history=max_history
    )

    # 获取蜡烛开盘价数据
    candle_opens = await exchange_public_data.Open(
        symbol=data_source_symbol, max_history=max_history
    )

    # 获取蜡烛收盘价数据
    candle_closes = await exchange_public_data.Close(
        symbol=data_source_symbol, max_history=max_history
    )

    # 获取蜡烛最高价数据
    candle_highs = await exchange_public_data.High(
        symbol=data_source_symbol, max_history=max_history
    )

    # 获取蜡烛最低价数据
    candle_lows = await exchange_public_data.Low(
        symbol=data_source_symbol, max_history=max_history
    )

    # 计算 HLC3
    candles_hlc3 = HLC3(
        candle_highs,
        candle_lows,
        candle_closes,
    )

    # 计算 OHLC4
    candles_ohlc4 = OHLC4(
        candle_opens,
        candle_highs,
        candle_lows,
        candle_closes,
    )

    # 根据用户选择的蜡烛源确定用户选择的蜡烛数据
    user_selected_candles = None
    if candle_source_name == PriceStrings.STR_PRICE_CLOSE.value:
        user_selected_candles = candle_closes
    if candle_source_name == PriceStrings.STR_PRICE_OPEN.value:
        user_selected_candles = await exchange_public_data.Open(
            ctx, symbol=data_source_symbol, max_history=max_history
        )
    if candle_source_name == PriceStrings.STR_PRICE_HIGH.value:
        user_selected_candles = candle_highs
    if candle_source_name == PriceStrings.STR_PRICE_LOW.value:
        user_selected_candles = candle_lows
    if candle_source_name == "hlc3":
        user_selected_candles = candles_hlc3
    if candle_source_name == "ohlc4":
        user_selected_candles = candles_ohlc4

    # 返回蜡烛数据
    return (
        candle_closes,
        candle_highs,
        candle_lows,
        candles_hlc3,
        candles_ohlc4,
        user_selected_candles,
        candle_times,
    )
