import numpy as np
import pandas as pd
from lib import encoding as en, outliers as out, summary as sum, graphic as gra
from lib import outliers as out, summary as sum, encoding as en
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)

df = pd.read_csv('dataset/ecommerce.csv')
dff = pd.read_csv('dataset/sellers.csv')

df.head()

df = pd.merge(df, dff, on='seller_id')

df.drop(['customer_unique_id', 'customer_zip_code_prefix', 'payment_sequential', 'review_comment_title',
         'product_name_length',
         'product_description_length', 'seller_zip_code_prefix'
         ], inplace=True, axis=1)

df.isnull().sum().sum()

df.dropna(subset=['product category', 'product_photos_qty', 'product_weight_g', 'product_length_cm'
    , 'product_height_cm', 'product_width_cm'], inplace=True)

out.for_check(df, df.columns)

low_th, up_th = out.outlier_thresholds(df, column=['freight_value'], q1=0.25, q3=0.75)

df[df['freight_value']>102.63].shape

df.describe().T

df.sort_values(by='payment_value', ascending=True).tail(30)

df['order_id'].value_counts()

df = df.drop_duplicates(subset=['order_id'])

df.isnull().sum()

df.shape

df.head()

gra.plot_numerical_col(df, num_cols=['price', 'payment_value','freight_value'])


df = out.replace_all_outliers(df, ['price', 'payment_value','freight_value','product_weight_g','product_width_cm'])

df.shape

out.for_check(df, df.columns)
