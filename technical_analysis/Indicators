import pandas as pd
import ta
import numpy as np


def calcular_macd(columna, signal_type: str, lenta=26, rapida=13, senal=9):
    MACD = ta.trend.MACD(columna, window_slow=lenta, window_fast=rapida, window_sign=senal)

    macd = MACD.macd()
    macd_signal = MACD.macd_signal()

    if signal_type == "BUY":
        return pd.DataFrame({"MACD": np.where(macd > macd_signal, True, False)})

    return pd.DataFrame({"MACD": np.where(macd < macd_signal, True, False)})


def calcular_sma(datos, columna, signal_type: str, window_short=5, window_long=15):
    short_sma = ta.trend.SMAIndicator(columna, window=window_short)
    long_sma = ta.trend.SMAIndicator(columna, window=window_long)
    datos = datos.copy()
    datos["short_sma"] = short_sma.sma_indicator()
    datos["long_sma"] = long_sma.sma_indicator()

    datos["buy_signal_sma"] = datos.short_sma < datos.long_sma
    datos["sell_signal_sma"] = datos.short_sma > datos.long_sma

    if signal_type == "BUY":
        return pd.DataFrame({"SMA": datos["buy_signal_sma"].shift(-1) & datos["sell_signal_sma"]})

    return pd.DataFrame({"SMA": datos["sell_signal_sma"].shift(-1) & datos["buy_signal_sma"]})


def calcular_bbands(datos, columna, signal_type: str, window=20, window_dev=2):
    BBANDS = ta.volatility.BollingerBands(columna, window=window, window_dev=window_dev)
    bband_h = BBANDS.bollinger_hband()
    bband_l = BBANDS.bollinger_lband()
    bband_ma = BBANDS.bollinger_mavg()

    bband_h_i = BBANDS.bollinger_hband_indicator()
    bband_l_i = BBANDS.bollinger_lband_indicator()

    datos = datos.copy()
    datos["Low_bband"] = bband_l
    datos["High_bband"] = bband_h
    datos["Media_bband"] = bband_ma

    datos["signal_High_bband"] = np.where(bband_h_i == 1, True, False)
    datos["signal_Low_bband"] = np.where(bband_l_i == 1, True, False)

    if signal_type == "BUY":
        return pd.DataFrame({"BB": np.where(datos["signal_Low_bband"] == True, True, False)})

    return pd.DataFrame({"BB": np.where(datos["signal_High_bband"] == True, True, False)})


def calcular_rsi(columna, signal_type: str, window=14, lower_bnd=35, upper_bnd=65):
    RSI_Indicator = ta.momentum.RSIIndicator(columna, window=window)
    rsi = RSI_Indicator.rsi()

    if signal_type == "BUY":
        return pd.DataFrame({"RSI": rsi < lower_bnd})

    return pd.DataFrame({"RSI": rsi > upper_bnd})



strats_dict = {
    "MACD": {
        "num_params": 3,
        "names": ["lenta", "rapida", "senal"],
        "bnds": ((16, 45), (1, 15), (5, 15))
    },
    "SMA": {
        "num_params": 2,
        "names": ["window_short", "window_long"],
        "bnds": ((1, 10), (11, 15))
    },
    "BBANDS": {
        "num_params": 2,
        "names": ["window", "window_dev"],
        "bnds": ((5, 40), (1, 3))
    },
    "RSI": {
        "num_params": 3,
        "names": ["window", "lower_bnd", "upper_bnd"],
        "bnds": ((5, 90), (1, 49), (50, 95))
    

    }}
