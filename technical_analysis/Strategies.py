from Indicators import strats_dict, calcular_macd, calcular_sma, calcular_rsi, calcular_bbands
import pandas as pd


def gen_signals(datos: pd.DataFrame, strat: list[int], signal_type: str, *args) -> pd.DataFrame:
    ct = 0
    strat_args = args
    macd, sma, bbands, rsi, pvo, estocastico = strat

    signals = pd.DataFrame()

    if macd:
        lenta = int(strat_args[ct])
        rapida = int(strat_args[ct + 1])
        senal = int(strat_args[ct + 2])
        ct += strats_dict["MACD"]["num_params"]

        if signal_type == "BUY":
            signals["MACD_BUY"] = calcular_macd(datos.Close, signal_type="BUY", lenta=lenta, rapida=rapida, senal=senal)
        else:
            signals["MACD_SELL"] = calcular_macd(datos.Close, signal_type="SELL", lenta=lenta, rapida=rapida,
                                                 senal=senal)

    if sma:
        window_short = int(strat_args[ct])
        window_long = int(strat_args[ct + 1])
        # print(window_short, window_long)
        ct += strats_dict["SMA"]["num_params"]

        if signal_type == "BUY":
            signals["SMA_BUY"] = calcular_sma(datos, datos.Close, signal_type="BUY", window_short=window_short,
                                              window_long=window_long)
        else:
            signals["SMA_SELL"] = calcular_sma(datos, datos.Close, signal_type="SELL", window_short=window_short,
                                               window_long=window_long)

    if bbands:
        window = int(strat_args[ct])
        window_dev = int(strat_args[ct + 1])
        ct += strats_dict["BBANDS"]["num_params"]

        if signal_type == "BUY":
            signals["SMA_BUY"] = calcular_bbands(datos, datos.Close, signal_type="BUY", window=window,
                                                 window_dev=window_dev)
        else:
            signals["SMA_SELL"] = calcular_bbands(datos, datos.Close, signal_type="SELL", window=window,
                                                  window_dev=window_dev)

    if rsi:
        window = int(strat_args[ct])
        lower_bound = int(strat_args[ct + 1])
        upper_bound = int(strat_args[ct + 2])
        ct += strats_dict["RSI"]["num_params"]

        if signal_type == "BUY":
            signals["MACD_BUY"] = calcular_rsi(datos.Close, signal_type="BUY", window=window, lower_bnd=lower_bound,
                                               upper_bnd=upper_bound)
        else:
            signals["MACD_SELL"] = calcular_rsi(datos.Close, signal_type="SELL", window=window, lower_bnd=lower_bound,
                                                upper_bnd=upper_bound)

   
