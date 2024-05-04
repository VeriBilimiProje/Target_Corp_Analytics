import numpy as np
import pandas as pd
from lib import encoding as en, outliers as out, summary as sum, graphic as gra
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

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

max_value_count = df.seller_id.value_counts().max()
seller = df.seller_id.value_counts().to_dict()
seller_popularity = []
for _id in df.seller_id.values:
    seller_popularity.append(seller[_id])
df['seller_popularity'] = seller_popularity

df['seller_popularity'] = pd.cut(df['seller_popularity'], bins=[0, 300, 1000, np.inf], labels=['C', 'B', 'A'])

df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)

df.drop_duplicates(subset=['review_id'], keep='first', inplace=True)

df.drop(df[df['payment_installments'] == 0].index, inplace=True)

df['estimated_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
df['ships_in'] = (df['shipping_limit_date'] - df['order_purchase_timestamp']).dt.days

df['shipping_days'] = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.days

df['answer_diff'] = (df['review_answer_timestamp'] - df['review_creation_date']).dt.total_seconds() / 86400

df['customer_wait_day'] = (df['order_delivered_customer_date'] - df[
    'order_purchase_timestamp']).dt.total_seconds() / 86400

df['quantity'] = (df['payment_value'] / (df['price'] + df['freight_value'])).astype(int)

df['quantity'].replace(0, 1, inplace=True)
df['discount'] = ((df['freight_value'] + df['price']) * df['quantity']) - df['payment_value']

df['delay_time'] = (df['order_estimated_delivery_date'] - df[
    'order_delivered_customer_date']).dt.total_seconds() / 86400

df.drop(['order_status'], axis=1, inplace=True)

df['season'] = df['order_purchase_timestamp'].dt.month

label = ['q1', 'q2', 'q3', 'q4']

df['season'] = pd.qcut(df['season'], 4, label)

df['year'] = df['order_purchase_timestamp'].dt.year

seller_review_score = df.groupby('seller_id')['review_score'].mean()

df['seller_review_score'] = df['seller_id'].map(seller_review_score * 2)

df.drop(df[df['year'] == 2016].index, inplace=True)

df[['payment_type']] = en.rare_encoder(df[['payment_type']], 0.06)
df[['product category']] = en.rare_encoder(df[['product category']], 0.02)

df.drop(df[df['shipping_days'] <= 0].index, inplace=True)

df.drop(df[df['product_weight_g'] == 0].index, inplace=True)

df.drop(df[df['distance_km'] == 0].index, inplace=True)

df.drop(df[(df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days < 0].index, inplace=True)

df.drop(df[(df['order_delivered_carrier_date'] - df['order_approved_at']).dt.days < 0].index, inplace=True)

df.drop(df[df['review_score'] == 3].index, inplace=True)

df['customer_wait_day'] = pd.cut(df['customer_wait_day'], bins=[0, 8, 16, 25, 40, 61],
                                 labels=['Very_Fast', 'Fast', 'Neutral', 'Slow', 'Worst'])

df['estimated_days'] = pd.cut(df['estimated_days'], bins=[0, 8, 16, 25, 40, 61],
                              labels=['Very_Fast', 'Fast', 'Neutral', 'Slow', 'Worst'])

df['ships_in'] = pd.cut(df['ships_in'], bins=[0, 4, 8, 16, 28, 61],
                        labels=['Very_Fast', 'Fast', 'Neutral', 'Slow', 'Worst'])

encoded_class = {1: 'Not Satisfied',
                 2: 'Not Satisfied',
                 3: 'Not Satisfied',
                 4: 'Satisfied',
                 5: 'Satisfied'}

df['review_score'] = df['review_score'].map(encoded_class)

df.head()

df.isnull().sum()

########################################################################
# MODELS
########################################################################

df_glad = df[
    ['review_score', 'price', 'freight_value', 'payment_type', 'payment_installments', 'payment_value',
     'customer_wait_day', 'seller_review_score', 'delay_time', 'distance_km', 'seller_popularity', 'discount'
     ]]

result = out.grab_col_names(df_glad)
cat_cols, num_cols = result[0], result[1]

# sum.correlation_matrix(df_glad,num_cols)
#
# sum.cat_summary(df_glad,cat_cols)
#
# sum.target_summary_with_cat(df_glad,'review_score',cat_cols)
#
for i in num_cols:
    sum.target_summary_with_num(df_glad, 'review_score', i)

sum.cat_summary(df_glad,cat_cols)

cols_list = ['price', 'freight_value', 'payment_value',
             'seller_review_score', 'delay_time', 'distance_km', 'discount']

df_glad = out.replace_all_outliers(df_glad, cols_list)

df_glad['price'] = np.log(df_glad['price'])
df_glad['payment_value'] = np.log(df_glad['payment_value'])
df_glad['seller_review_score'] = np.log(df_glad['seller_review_score'])
df_glad['distance_km'] = np.log(df_glad['distance_km'])

df_glad['delay_time'].fillna(-14, inplace=True)
df_glad['distance_km'].fillna(428, inplace=True)
df_glad['customer_wait_day'].fillna('Worst', inplace=True)
df_glad.dropna(inplace=True)

df_glad = en.one_hot_encoder(df_glad, ['payment_type', 'customer_wait_day', 'seller_popularity'], drop_first=True)

df_glad = en.label_encoder(df_glad, 'review_score')

rs = RobustScaler()
l = [col for col in df_glad.columns if col not in ['review_score']]
df_glad[l] = rs.fit_transform(df_glad[l])

y = df_glad['review_score']
X = df_glad.drop(columns=['review_score'], axis=1)

df_glad.head()

########################################################################
# Model 1) Linear Regression
########################################################################


l_model = LogisticRegression().fit(X, y)

param = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 30, 40]}

LR = GridSearchCV(l_model, param, cv=3, refit=False, return_train_score=True, scoring='roc_auc')
LR.fit(X, y)

l_model = l_model.set_params(**LR.best_params_)

cv_results = cross_validate(l_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8248765502283758
# 0.8808304779078979
cv_results['test_f1'].mean()
# 0.9037088841310366
# 0.9327284353212955
cv_results['test_roc_auc'].mean()
# 0.5484968244473337
# 0.7990957279062073


RocCurveDisplay.from_estimator(l_model, X, y)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

df_glad['review_score'].value_counts()

########################################################################
# Model 2) Random Forest Regression
########################################################################

forest_reg = RandomForestClassifier(random_state=42)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "sqrt", "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

x = {'max_depth': 5,
     'max_features': 7,
     'min_samples_split': 5,
     'n_estimators': 100}

rf_best_grid = GridSearchCV(forest_reg, rf_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

forest_reg = forest_reg.set_params(**x)

forest_reg = forest_reg.fit(X, y)

cv_results = cross_validate(forest_reg, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8923785685665253
cv_results['test_f1'].mean()
# 0.9408376843109322
cv_results['test_roc_auc'].mean()
# 0.7621464859119431

########################################################################
# Model 3) CART
########################################################################
cart_model = DecisionTreeClassifier(random_state=1)

cart_model.fit(X, y)

cv_results = cross_validate(cart_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7034600008390696
cv_results['test_f1'].mean()
# 0.8063544245071561
cv_results['test_roc_auc'].mean()
# 0.5915474978828692
########################################################################
# Model 3) XGBOOST
########################################################################
xg_model = XGBClassifier(objective='reg:squarederror')

xg_model = xg_model.fit(X, y)

cv_results = cross_validate(xg_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.5927073446043444
cv_results['test_f1'].mean()
# 0.6626728520667822
cv_results['test_roc_auc'].mean()
# 0.6072938920058083

########################################################################
# Model 3) CATBOOST
########################################################################

cat_model = CatBoostClassifier().fit(X, y)

cv_results = cross_validate(cat_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.611522356744368
cv_results['test_f1'].mean()
# 0.6909165576165092
cv_results['test_roc_auc'].mean()
# 0.6358588778104648


########################################################################
# Model 3) KNN
########################################################################

knn_model = KNeighborsClassifier()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_final = knn_model.set_params(**knn_gs_best.best_params_)

cv_results = cross_validate(knn_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8812944167620393
cv_results['test_f1'].mean()
# 0.9334593883193723
cv_results['test_roc_auc'].mean()
# 0.7467817359651997

########################################################################
# Model 3) LightGBM
########################################################################
lgbm_model = LGBMClassifier(random_state=17)

lgbm_model = lgbm_model.fit(X, y)

cv_results = cross_validate(lgbm_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7241318072463834
cv_results['test_f1'].mean()
# 0.8099357471188142
cv_results['test_roc_auc'].mean()
# 0.6684717949138276

########################################################################
# Model 3) Adaboost
########################################################################
ada_reg = AdaBoostClassifier(random_state=42)
ada_reg.get_params()

adab_params = {"learning_rate": [0.0001, 0.001, 0.01, 0.1, 1, 10,],
               "n_estimators": [50,100,200,300, 500]}

ada_best_grid = GridSearchCV(ada_reg, adab_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

ada_reg = ada_reg.set_params(**ada_best_grid.best_params_)

cv_results = cross_validate(ada_reg, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8923785685665253
cv_results['test_f1'].mean()
# 0.9408376843109322
cv_results['test_roc_auc'].mean()

########################################################################
# Model 3) GBM
########################################################################
gbm_reg = GradientBoostingClassifier(random_state=42)
gbm_reg.get_params()

gbm_params = {"learning_rate": [0.0001, 0.001, 0.01, 0.1, 1, 10,],
              "max_depth": [3, 5,8,10],
              "n_estimators": [100,500, 1000],
              "subsample": [1, 0.5, 0.7,2]}

gbm_best_grid = GridSearchCV(gbm_reg, gbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

gbm_reg = gbm_reg.set_params(**gbm_best_grid.best_params_)

cv_results = cross_validate(gbm_reg, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8923785685665253
cv_results['test_f1'].mean()
# 0.9408376843109322
cv_results['test_roc_auc'].mean()


########################################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(forest_reg, X)

# df.drop(df[df['review_score'] == 5].head(30000).index, inplace=True)

# indices_to_drop = df[df['review_score'] == 5].index
#
# num_rows_to_drop = 30000
#
# random_indices = np.random.choice(indices_to_drop, min(num_rows_to_drop, len(indices_to_drop)), replace=False)
#
# df.drop(random_indices, inplace=True)
