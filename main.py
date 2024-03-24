import numpy as np
import pandas as pd
from lib import encoding as en, outliers as out, summary as sum
from lib import outliers as out, summary as sum, encoding as en
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option('display.max_column', None)
pd.set_option('display.width', 500)

df = pd.read_csv('dataset/ecommerce.csv')

df.head()


