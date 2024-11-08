from model.review_prediction import research as re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import joblib

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)

df = pd.read_csv('datasets/dataset.csv')

X, y = re.data_prep(df)
lr_reg = LogisticRegression().fit(X, y)

cv_results = cross_validate(lr_reg, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
cv_results['test_accuracy'].mean()

joblib.dump(lr_reg, "deployment/review_model.pkl")

re.base_models(X, y)

re.hyperparameter_optimization(X, y)

#           accuracy        f1   roc_auc
# LR        0.882363  0.933584  0.785310
# KNN       0.872191  0.927221  0.724845 || 0.883279  0.934255  0.771098
# SVC       0.885738  0.935655  0.740156
# CART      0.785999  0.870041  0.643320 || 0.890884  0.938397  0.760589
# RF        0.883481  0.933711  0.776550 || 0.891407  0.938665  0.807641
# Adaboost  0.891181  0.938322  0.802404 || 0.891620  0.938689  0.811354
# GBM       0.890230  0.937701  0.808648
# XGBoost   0.871229  0.926038  0.769873 || 0.890896  0.938410  0.807098
# LightGBM  0.881449  0.932388  0.793221 || 0.886428  0.935597  0.801090
# CatBoost  0.875031  0.928310  0.782372


#           accuracy        f1   roc_auc
# LR        0.881353  0.933096  0.794200
# KNN       0.883279  0.934255  0.771098
# SVC       0.885738  0.935655  0.740156
# CART      0.890884  0.938397  0.760589
# RF        0.891407  0.938665  0.807641
# Adaboost  0.891620  0.938689  0.811354
# GBM       0.890230  0.937701  0.808648
# XGBoost   0.890896  0.938410  0.807098
# LightGBM  0.886428  0.935597  0.801090
# CatBoost  0.875031  0.928310  0.782372
