import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import optuna
import xgboost as xgb
import pickle


def create_dataset():
    df = pd.read_csv("../data/aapl_5m_train.csv")
    df[f'T_Minus_1'] = df['Close'].shift(1)
    df[f'T_Minus_2'] = df['Close'].shift(2)
    df[f'T_Minus_3'] = df['Close'].shift(3)
    df["T_Plus_5"] = df['Close'].shift(-5)
    df = df.dropna()  
    return df

def model_data(df, trade):
    x = df[['Close', 'T_Minus_1', 'T_Minus_2', 'T_Minus_3']]
    y = df['Close'] < df['T_Plus_5']
    
    if trade == "short":
        y = df['Close'] > df['T_Plus_5']
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1, shuffle=False)
    return X_train, X_test, y_train, y_test

def logistic_regression(tuned_hyper_params):
    model = LogisticRegression(random_state = 20, C=tuned_hyper_params["C"],fit_intercept=tuned_hyper_params["fit_intercept"], l1_ratio=tuned_hyper_params["l1_ratio"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    return f1, model

def objective_log_reg(trial):
    C = trial.suggest_loguniform('C', 1e-5, 1e5)
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
    model = LogisticRegression(C=C,fit_intercept=fit_intercept,l1_ratio=l1_ratio,random_state=20)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    return f1

def objective_svc(trial):
    C = trial.suggest_loguniform('C', 1e-5, 1e5)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    gamma = trial.suggest_loguniform('gamma', 1e-5, 1e1)
    svc_model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=56, max_iter=2_000)
    svc_model.fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    return f1

def svc(tuned_hyper_params):
    model = SVC(random_state = 56, max_iter=2_000, C=tuned_hyper_params["C"], gamma=tuned_hyper_params["gamma"], kernel=tuned_hyper_params["kernel"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    return f1, model

def objective_xgboost(trial):
     params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'max_leaves': trial.suggest_int('max_leaves', 5, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }
     
     dtrain = xgb.DMatrix(X_train, label=y_train)
     dtest = xgb.DMatrix(X_test, label=y_test)
     num_rounds = 100
     model = xgb.train(params, dtrain, num_rounds)
     y_pred = model.predict(dtest)
     y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
     f1 = f1_score(y_test, y_pred_binary)
     return f1
 
def xgboost(best_params):
     params = {
        'n_estimators': best_params["n_estimators"],
        'max_depth': best_params["max_depth"],
        'max_leaves': best_params["max_leaves"],
        'learning_rate': best_params["learning_rate"],
        'booster': best_params["booster"],
        'gamma': best_params["gamma"],
        'reg_alpha': best_params["reg_alpha"],
        'reg_lambda': best_params["reg_lambda"],
    }
     
     dtrain = xgb.DMatrix(X_train, label=y_train)
     dtest = xgb.DMatrix(X_test, label=y_test)
     num_rounds = 100
     model = xgb.train(params, dtrain, num_rounds)
     y_pred = model.predict(dtest)
     y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
     f1 = f1_score(y_test, y_pred_binary)
     return f1, model

dataset = create_dataset()
X_train, X_test, y_train, y_test = model_data(dataset, "short")

#logistic_regression_study = optuna.create_study(direction='maximize')
#logistic_regression_study.optimize(objective_log_reg, n_trials=10)
#f1_log_reg, log_model = logistic_regression(logistic_regression_study.best_params)

svc_study = optuna.create_study(direction='maximize')
svc_study.optimize(objective_svc, n_trials=10)
f1_svc, svc_model = svc(svc_study.best_params)

#xg_study = optuna.create_study(direction='maximize')
#xg_study.optimize(objective_xgboost, n_trials=50)
#f1_xg, xg_model = xgboost(xg_study.best_params)  

print(f1_svc)

#Guardar modelos con pickle
with open('modelo_svc_5m_short.pkl', 'wb') as file:
    pickle.dump(svc_model, file)