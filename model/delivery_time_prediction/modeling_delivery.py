from model.delivery_time_prediction import research_delivery as re
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import joblib
from catboost import CatBoostRegressor


pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)

df = pd.read_csv('datasets/dataset.csv')

X, y = re.data_prep(df)

catb_reg = CatBoostRegressor(verbose=False).fit(X, y)

re.base_models(X, y)

re.hyperparameter_optimization(X, y)

joblib.dump(catb_reg, "deployment/logistic_model.pkl")

#               rmse  r2_score
# LR        4.976928  0.720284
# KNN       5.319954  0.680286
# SVC       4.876366  0.731605
# CART      3.136245  0.891256
# RF        2.286691  0.939461
# Adaboost  9.600065 -0.238313
# GBM       2.484970  0.930158
# XGBoost   1.631821  0.969866
# LightGBM  1.710070  0.966815
# CatBoost  1.164257  0.984683

#               rmse  r2_score
# LR        4.744349  0.745822
# KNN       5.719373  0.630438
# CART      2.207109  0.941249
# RF        1.533162  0.973467
# GBM       1.732820  0.966031
# XGBoost   1.488365  0.974814 || 1.316642  0.980264
# LightGBM  1.396390  0.977901 || 1.344612  0.979495
# CatBoost  1.176823  0.983966