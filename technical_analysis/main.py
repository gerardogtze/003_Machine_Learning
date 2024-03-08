
import numpy as np
import pandas as pd
import ta
from Indicators import calcular_macd,calcular_sma,calcular_rsi,calcular_bbands

# Cargar datos de entrenamiento desde el archivo CSV
train_data = pd.read_csv('C:/Users/Santiago/Desktop/ITESO/9no Semestre/Microestructuras de Trading/002TechnicalAnalysis/files/aapl_5m_train.csv')

# Cargar datos de validación desde el archivo CSV
validation_data = pd.read_csv('C:/Users/Santiago/Desktop/ITESO/9no Semestre/Microestructuras de Trading/002TechnicalAnalysis/files/aapl_5m_validation.csv')


calcular_macd(train_data,train_data.Close)
calcular_sma(train_data,train_data.Close)
calcular_bbands(train_data,train_data.Close)
calcular_estocastico(train_data,train_data.Close)
calcular_rsi(train_data,train_data.Close)
calculate_pvo_signals(train_data,train_data.Close)


n = 4
nums = list(range(1, 2 ** n))
combinations = list(map(lambda x: [int(x) for x in "{0:04b}".format(n)], nums))
print(combinations)

