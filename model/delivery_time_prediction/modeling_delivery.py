from model.delivery_time_prediction import research_delivery as re
import pandas as pd

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)

df = pd.read_csv('datasets/dataset.csv')

X, y = re.data_prep(df)

re.base_models(X, y)

re.hyperparameter_optimization(X, y)
