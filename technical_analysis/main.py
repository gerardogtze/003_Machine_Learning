import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
#
data = pd.read_csv("aapl_5m_train.csv")
data = pd.read_csv("aapl_1m_train.csv")
data = pd.read_csv("aapl_1h_train.csv")
data.head()x
#
rsi_data = ta.momentum.RSIIndicator(close=data.Close, window=14)
data["RSI"] = rsi_data.rsi() #Shift tab nos da la info de la función
data = data.dropna()
data.head()
#
fig, axs = plt.subplots(2, 1, figsize=(14,8)) #Crear la figura

#Price chart
axs[0].plot(data.Close[:214])
#Oscillator
axs[1].plot(data.RSI[:214])
axs[1].plot([0, 214], [70, 70], 'r--', label="Upper Threshold")
axs[1].plot([0, 214], [30, 30], 'g--', label="Lower Threshold")
plt.legend()
plt.show()
#
class Operation:
    def __init__(self, operation_type, bought_at, timestamp, n_shares,
                stop_loss, take_profit):
        self.operation_type = operation_type
        self.bought_at = bought_at
        self.timestamp = timestamp
        self.n_shares = n_shares
        self.sold_at = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit

#
#Primera clase
cash = 1_000_000
active_operations = []
comision = 0.00125
strategy_value = [1_000_000]

for i, row in data.iterrows():
    #Cerrar las operaciones
    temp_operations = []
    for op in active_operations:
        #Cerrar las operaciones de perdida
        if op.stop_loss > row.Close:
            cash += row.Close * (1 - comision)
        #Cerrar las operaciones con ganancia
        elif op.take_profit < row.Close:
            cash += row.Close * (1 - comision)
        else:
            temp_operations.append(op)
    active_operations = temp_operations
    
    #Comprobar si tenemos suficiente dinero para comprar
    if cash > row.Close * (1 + comision):
        #Señal de compra
        if row.Close > row.Open:
            active_operations.append(Operation(operation_type ="long",
                                              bought_at = row.Close,
                                              timestamp = row.Timestamp,
                                              n_shares= 1,
                                              stop_loss=row.Close * 0.95,
                                              take_profit=row.Close * 1.05)) #Guarda los datos en Operation
            cash -= row.Close * (1 + comision)
    #Calcular el valor de las posiciones abiertas
    total_value = len(active_operations) * row.Close
    strategy_value.append(cash + total_value)

#RSI Strategy
#Cambiamos por RSI
cash = 1_000_000
active_operations = []
comision = 0.00125
strategy_value = [1_000_000]
n_shares = 40 #Titulos que voy a comprar cada que haga una operación

for i, row in data.iterrows():
    #Cerrar las operaciones
    temp_operations = []
    for op in active_operations:
        #Cerrar las operaciones de perdida
        if op.stop_loss > row.Close:
            cash += row.Close * op.n_shares *(1 - comision) #multiplicamos por n_shares porque ese dinero perdimos por cada titulo
        #Cerrar las operaciones con ganancia
        elif op.take_profit < row.Close:
            cash += row.Close * op.n_shares * (1 - comision)
        else:
            temp_operations.append(op)
    active_operations = temp_operations
    
    #Comprobar si tenemos suficiente dinero para comprar
    if cash > row.Close * n_shares * (1 + comision):
        #Señal de compra, if RSI < 30, buy...
        if row.RSI < 30:
            active_operations.append(Operation(operation_type ="long",
                                              bought_at = row.Close,
                                              timestamp = row.Timestamp,
                                              n_shares= n_shares,
                                              stop_loss=row.Close * 0.95,
                                              take_profit=row.Close * 1.05)) #Guarda los datos en Operation
            cash -= row.Close * n_shares * (1 + comision)
        else:
            print("No hay dinero disponible")
    #Calcular el valor de las posiciones abiertas
    total_value = len(active_operations) * row.Close * n_shares
    strategy_value.append(cash + total_value)

#
plt.figure(figsize=(12,8))
plt.plot(strategy_value)
plt.title("Primera estrategia de trading")
plt.show()

#SMA Strategy

short_ma = ta.trend.SMAIndicator(data.Close, window=5)
long_ma = ta.trend.SMAIndicator(data.Close, window=21)
data["SHORT_SMA"] = short_ma.sma_indicator()
data["LONG_SMA"] = long_ma.sma_indicator()
data = data.dropna()
data.head()
#
plt.figure(figsize=(12, 6))
plt.plot(data.Close[:250], label="Price")
plt.plot(data.SHORT_SMA[:250], label="SMA(5)")
plt.plot(data.LONG_SMA[:250], label="SMA(21)")
plt.legend()
plt.show()
#
class Operation:
    def __init__(self, operation_type, bought_at, timestamp, n_shares,
                stop_loss, take_profit):
        self.operation_type = operation_type
        self.bought_at = bought_at
        self.timestamp = timestamp
        self.n_shares = n_shares
        self.sold_at = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit

#
data.iloc[0].SHORT_SMA

##
data.iloc[0].LONG_SMA

#
#Cambiamos por RSI
cash = 1_000_000
active_operations = []
comision = 0.00125
strategy_value = [1_000_000]
n_shares = 40 #Titulos que voy a comprar cada que haga una operación

sma_sell_signal = data.iloc[0].LONG_SMA > data.iloc[0].SHORT_SMA
sma_buy_signal = data.iloc[0].LONG_SMA < data.iloc[0].SHORT_SMA

for i, row in data.iterrows():
    #Cerrar las operaciones
    temp_operations = []
    for op in active_operations:
        #Cerrar las operaciones de perdida
        if op.stop_loss > row.Close:
            cash += row.Close * op.n_shares *(1 - comision) #multiplicamos por n_shares porque ese dinero perdimos por cada titulo
        #Cerrar las operaciones con ganancia
        elif op.take_profit < row.Close:
            cash += row.Close * op.n_shares * (1 - comision)
        else:
            temp_operations.append(op)
    active_operations = temp_operations
    
    #Comprobar si tenemos suficiente dinero para comprar
    if cash > row.Close * n_shares * (1 + comision):
        #Ver si la señal de compra cambió
        if (row.LONG_SMA < row.SHORT_SMA) and sma_buy_signal == False:
            sma_buy_signal = True
            #Comprar
            active_operations.append(Operation(operation_type ="long",
                                              bought_at = row.Close,
                                              timestamp = row.Timestamp,
                                              n_shares= n_shares,
                                              stop_loss=row.Close * 0.95,
                                              take_profit=row.Close * 1.05)) #Guarda los datos en Operation
            cash -= row.Close * n_shares * (1 + comision)
        elif row.LONG_SMA > row.SHORT_SMA:
            sma_buy_signal = False
            
        #Señal de compra, if RSI < 30, buy...
        #if row.RSI < 30:
        #    active_operations.append(Operation(operation_type ="long",
        #                                      bought_at = row.Close,
        #                                      timestamp = row.Timestamp,
        #                                     n_shares= n_shares,
        #                                      stop_loss=row.Close * 0.95,
        #                                      take_profit=row.Close * 1.05)) #Guarda los datos en Operation
        #    cash -= row.Close * n_shares * (1 + comision)
    else:
        print("No hay dinero disponible")
    #Calcular el valor de las posiciones abiertas
    total_value = len(active_operations) * row.Close * n_shares
    strategy_value.append(cash + total_value)
        
#
plt.figure(figsize=(12,4))
plt.plot(strategy_value)
plt.title("Primera estrategia de trading")
plt.show()
