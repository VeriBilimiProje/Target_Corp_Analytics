import numpy as np
import pandas as pd
from lib import encoding as en, outliers as out, summary as sum, graphic as gra
from lib import outliers as out, summary as sum, encoding as en
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.impute import KNNImputer
import joblib

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)

df = pd.read_csv('datasets/dataset.csv')

########################################################################
# DATA ANALYSIS
########################################################################

df.shape
# (115609, 31)

df.nunique()

df.head()

result = out.grab_col_names(df)

cat_cols, num_cols = result[0], result[1]

gra.plot_numerical_col(df, num_cols=num_cols, plot_type='kde')

gra.plot_categoric_col(df, cat_cols)

df.describe().T

########################################################################
# MISSING DATA ANALYSIS & OUTLIERS DATA ANALYSIS
########################################################################
df.head()

df.shape
df.isnull().sum()

out.for_check(df, df.columns)

df = out.remove_all_outliers(df, ['price', 'payment_value', 'freight_value', 'product_weight_g', 'product_width_cm',
                                  'payment_installments', 'payment_value', 'distance_km'])

df.drop(df[df['order_status'] == 'canceled'].index, inplace=True)

sum.cat_summary(df, cat_cols)

df.info()

df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)

df.drop_duplicates(subset=['review_id'], keep='first', inplace=True)

df.drop(df[df['payment_installments'] == 0].index, inplace=True)

########################################################################
# FEATURE EXTRACTION
########################################################################


list_of_date_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
                        'order_delivered_customer_date', 'order_estimated_delivery_date',
                        'review_creation_date', 'review_answer_timestamp', 'shipping_limit_date',
                        'order_estimated_delivery_date']

for col in list_of_date_columns:
    try:
        df[col] = df[col].astype('datetime64[ns]')
    except ValueError:
        df['review_creation_date'] = pd.to_datetime(df['review_creation_date'].str[:10], format='%Y-%m-%d')
        print(f'{col} dont')

df.head()

df['shipping_days'] = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.days

df['answer_diff'] = (df['review_answer_timestamp'] - df['review_creation_date']).dt.total_seconds() / 86400

df['approval_time'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / 60

df['customer_wait_day'] = (df['order_delivered_customer_date'] - df[
    'order_purchase_timestamp']).dt.total_seconds() / 86400

df['quantity'] = (df['payment_value'] / (df['price'] + df['freight_value'])).astype(int)

df['quantity'].replace(0, 1, inplace=True)

# df['purchase_weekday'] = df['order_purchase_timestamp'].dt.weekday
#
# df['purchase_weekday'] = df['purchase_weekday'].replace({5: 0, 6: 0, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1})
#
# special_days = {
#     'New Year': ['01-01'],
#     'Carnival': ['02-24', '02-25', '02-26'],
#     'Valentine\'s Day': ['06-12'],
#     'Children\'s Day': ['10-12'],
#     'Black Friday': ['11-27'],
#     'Christmas': ['12-25']
# }
#
# df['special_day'] = 0
#
# for event, dates in special_days.items():
#     for date in dates:
#         df.loc[df['order_purchase_timestamp'].dt.strftime('%m-%d') == date, 'special_day'] = 1

df['delay_time'] = (df['order_estimated_delivery_date'] - df[
    'order_delivered_customer_date']).dt.total_seconds() / 86400

df.drop(df[df['order_status'] == 'canceled'].index, inplace=True)

df.drop(['order_status'], axis=1, inplace=True)

df['season'] = df['order_purchase_timestamp'].dt.month

label = ['q1', 'q2', 'q3', 'q4']

df['season'] = pd.qcut(df['season'], 4, label)

df['year'] = df['order_purchase_timestamp'].dt.year

# earnings_by_year_and_season = df.groupby(['year', 'season'])['payment_value'].sum()
#
# total_earnings = df.apply(lambda row: earnings_by_year_and_season[(row['year'], row['season'])], axis=1)

# df['total_earn_quantile'] = total_earnings

# earnings_by_year_season_seller = df.groupby(['year', 'season', 'seller_id'])['payment_value'].sum()

# total_earning_by_seller = df.apply(
#     lambda row: earnings_by_year_season_seller.get((row['year'], row['season'], row['seller_id']), 0), axis=1)

# df['total_earn_quantile_by_seller'] = total_earning_by_seller

seller_review_score = df.groupby('seller_id')['review_score'].mean()

df['seller_review_score'] = df['seller_id'].map(seller_review_score * 2)

df.drop(df[df['year'] == 2016].index, inplace=True)

df[['payment_type', 'product category']] = en.rare_encoder(df[['payment_type', 'product category']], 0.06)

df.drop(df[df['shipping_days'] <= 0].index, inplace=True)

df.drop(df[df['shipping_days'] <= 0].index, inplace=True)

df.drop(df[df['product_weight_g'] == 0].index, inplace=True)

df.drop(df[df['approval_time'] == 0].index, inplace=True)

df.drop(df[df['distance_km'] == 0].index, inplace=True)

df.drop(df[(df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days < 0].index, inplace=True)

df.drop(df[(df['order_delivered_carrier_date'] - df['order_approved_at']).dt.days < 0].index, inplace=True)

encoded_class = {1: 'Not Satisfied',
                 2: 'Not Satisfied',
                 3: 'Not Satisfied',
                 4: 'Satisfied',
                 5: 'Satisfied'}

df['review_score'] = df['review_score'].map(encoded_class)

########################################################################
# MODELS
########################################################################

df_glad = df[
    ['review_score', 'price', 'freight_value', 'payment_type', 'payment_installments', 'payment_value',
     'approval_time', 'customer_wait_day', 'seller_review_score', 'product category', 'quantity', 'year', 'season',
     'delay_time', 'distance_km', 'cities_status'
     ]]

df_glad = out.remove_top_frequency(df_glad, 'freight_value', threshold=0.002)

df['price'] = np.log(df['price'])
df['payment_value'] = np.log(df['payment_value'])
df['approval_time'] = np.log(df['approval_time'])
df['customer_wait_day'] = np.log(df['customer_wait_day'])
df['seller_review_score'] = np.log(df['seller_review_score'])
df['quantity'] = np.log(df['quantity'])
df['year'] = np.log(df['year'])
df['distance_km'] = np.log(df['distance_km'])

result = out.grab_col_names(df_glad)

cat_cols, num_cols = result[0], result[1]

sum.cat_summary(df, cat_cols)

gra.plot_numerical_col(df_glad, num_cols=num_cols, plot_type='kde')

cols_list = ['price', 'freight_value', 'payment_value', 'approval_time',
             'customer_wait_day', 'seller_review_score', 'quantity', 'year', 'delay_time', 'distance_km']

df_glad = out.remove_all_outliers(df_glad, cols_list)

df_glad['approval_time'].fillna(2206, inplace=True)
df_glad['customer_wait_day'].fillna(40, inplace=True)
df_glad['delay_time'].fillna(-14, inplace=True)
df_glad.dropna(inplace=True)

# corr = df[num_cols].corr()
# cor_matrix = corr.abs()
# upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
# drop_list = [col for col in upper_triangle_matrix if any(upper_triangle_matrix[col] > 0.9)]
# sns.set(rc={'figure.figsize': (20, 20)})
# sns.heatmap(corr, cmap='RdBu')
# plt.show()

df_glad = en.one_hot_encoder(df_glad, ['payment_type', 'season', 'cities_status'], drop_first=True)

df_glad = en.label_encoder(df_glad, 'review_score')
df_glad = en.label_encoder(df_glad, 'product category')

rs = RobustScaler()
l = [col for col in df_glad.columns if col not in ['review_score']]
df_glad[l] = rs.fit_transform(df_glad[l])

y = df_glad['review_score']
X = df_glad.drop(columns=['review_score'], axis=1)

########################################################################
# Model 1) Linear Regression
########################################################################


l_model = LogisticRegression().fit(X, y)

joblib.dump(l_model, "deployment/log_reg.pkl")

cv_results = cross_validate(l_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7769250469005519
# 0.816921068648272
cv_results['test_f1'].mean()
# 0.8744067332593181
# 0.8914220997748533
cv_results['test_roc_auc'].mean()
# 0.5244540528559379
# 0.7384002359670425

df.shape
########################################################################
# Model 2) Random Forest Regression
########################################################################
from sklearn.ensemble import RandomForestClassifier

forest_reg = RandomForestClassifier(random_state=42)

forest_reg.get_params()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "sqrt", "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(forest_reg, rf_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

forest_reg = forest_reg.set_params(**rf_best_grid.best_params_)

forest_reg.fit(X, y)

cv_results = cross_validate(forest_reg, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7630486338152581
cv_results['test_f1'].mean()
# 0.8462710315551891
cv_results['test_roc_auc'].mean()
# 0.6653456313884493

########################################################################
# Model 3) CART
########################################################################
cart_model = DecisionTreeClassifier(random_state=1)

cart_model.fit(X, y)

cv_results = cross_validate(cart_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.6546813197307091
cv_results['test_f1'].mean()
# 0.7609614618648842
cv_results['test_roc_auc'].mean()
# 0.5737582127798337
########################################################################
# Model 3) XGBOOST
########################################################################
xg_model = XGBClassifier(objective='reg:squarederror')

xg_model.fit(X, y)

cv_results = cross_validate(xg_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.6838286996825176
cv_results['test_f1'].mean()
# 0.7539813138016057
cv_results['test_roc_auc'].mean()
# 0.6156301372796502

########################################################################
# Model 3) CATBOOST
########################################################################

cat_model = CatBoostClassifier().fit(X, y)

cv_results = cross_validate(cat_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.6649307839796682
cv_results['test_f1'].mean()
# 0.7309140355724563
cv_results['test_roc_auc'].mean()
# 0.6317152047313181


# df.drop(df[df['review_score'] == 5].head(30000).index, inplace=True)

# indices_to_drop = df[df['review_score'] == 5].index
#
# num_rows_to_drop = 30000
#
# random_indices = np.random.choice(indices_to_drop, min(num_rows_to_drop, len(indices_to_drop)), replace=False)
#
# df.drop(random_indices, inplace=True)
