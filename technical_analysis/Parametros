import numpy as np
from scipy.optimize import minimize
from Minimize_portafolio import minimize_portafolio
from Indicators import strats_dict
from Objective_function import objective_function


def opt(strat: list[int]) -> np.array:
    bounds = []

    macd, sma, bbands, rsi = strat
    strat_args = []

    if macd:
        bounds += strats_dict["MACD"]["bnds"]
        strat_args += [strats_dict["MACD"]["names"]]

    if sma:
        bounds += strats_dict["SMA"]["bnds"]
        strat_args += [strats_dict["SMA"]["names"]]

    if bbands:
        bounds += strats_dict["BBANDS"]["bnds"]
        strat_args += [strats_dict["BBANDS"]["names"]]

    if rsi:
        bounds += strats_dict["RSI"]["bnds"]
        strat_args += [strats_dict["RSI"]["names"]]


    # Convierte las restricciones en un formato adecuado para minimize
    bounds = tuple(bounds)
    print(bounds)

    # Inicializa X0 (valores iniciales para la optimización)
    X0 = [np.mean(bound) for bound in bounds]


    # Define una función objetivo que llame a minimize_portafolio con los argumentos adecuados
    #objective_fun = objective_function(strat_args)

    # Llama a minimize para encontrar los valores óptimos
    result = minimize(objective_fun, X0, method='Nelder-Mead', bounds=bounds, args=(strat_args,))

    # Devuelve los valores óptimos encontrados
    return result.x
