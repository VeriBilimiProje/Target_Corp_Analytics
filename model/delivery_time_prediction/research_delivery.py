import numpy as np
import pandas as pd
from lib import encoding as en, outliers as out
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.impute import KNNImputer
from lightgbm import LGBMRegressor

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)

df = pd.read_csv('datasets/dataset.csv')


def data_prep(df):
    ########################################################################
    # FEATURE ENGINEERRING
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

    ########################################################################
    # FEATURE EXTRACTION
    ########################################################################

    df['prepare_time'] = (df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days

    df['approved_time'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.days

    df['shipping_days'] = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.days

    df['cargo_score'] = df['distance_km'] / df['shipping_days']

    df['delivery_time'] = (df['order_delivered_customer_date'] - df[
        'order_purchase_timestamp']).dt.days

    df['quantity'] = (df['payment_value'] / (df['price'] + df['freight_value'])).astype(int)

    df['quantity'].replace(0, 1, inplace=True)

    df['season'] = df['order_purchase_timestamp'].dt.month

    df['month'] = df['order_purchase_timestamp'].dt.month

    label = ['q1', 'q2', 'q3', 'q4']

    df['season'] = pd.qcut(df['season'], 4, label)

    df['year'] = df['order_purchase_timestamp'].dt.year

    df['purchase_weekday'] = df['order_purchase_timestamp'].dt.day_name()

    df['special_day'] = 'normal'

    special_days = {
        'New Year': ['01-01'],
        'Carnival': ['02-24', '02-25', '02-26'],
        'Valentine\'s Day': ['06-12'],
        'Children\'s Day': ['10-12'],
        'Black Friday': ['11-27'],
        'Christmas': ['12-25']
    }
    for event, dates in special_days.items():
        for date in dates:
            df.loc[df['order_purchase_timestamp'].dt.strftime('%m-%d') == date, 'special_day'] = event

    ########################################################################
    # INCONSISTENT VALUES
    ########################################################################

    df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)

    df.drop_duplicates(subset=['review_id'], keep='first', inplace=True)

    df.drop(df[df['payment_installments'] == 0].index, inplace=True)

    df.drop(['order_status'], axis=1, inplace=True)

    df.drop(df[df['year'] == 2016].index, inplace=True)

    df.drop(df[df['shipping_days'] <= 0].index, inplace=True)

    df.drop(df[df['product_weight_g'] == 0].index, inplace=True)

    df.drop(df[df['distance_km'] == 0].index, inplace=True)

    df.drop(df[(df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days < 0].index, inplace=True)

    df.drop(df[(df['order_delivered_carrier_date'] - df['order_approved_at']).dt.days < 0].index, inplace=True)

    df['product_cm3'] = df['product_length_cm'] * df['product_width_cm'] * df['product_height_cm']

    ########################################################################
    # RARE ENCODING
    ########################################################################

    df[['customer_state', 'seller_state']] = (
        en.rare_encoder(df[['customer_state', 'seller_state']], 0.01))

    ########################################################################
    # FINAL DATASET
    ########################################################################

    df_final = df[['product_weight_g', 'payment_value', 'distance_km', 'cities_status',
                   'quantity', 'season', 'year', 'product_cm3', 'purchase_weekday', 'special_day', 'month',
                   'delivery_time',
                   'prepare_time', 'cargo_score', 'seller_state',
                   'customer_state'
                   ]]

    ########################################################################
    # MISSING VALUES
    ########################################################################
    imputer = KNNImputer(n_neighbors=10)
    df_final[['distance_km']] = (imputer.fit_transform(df_final[['distance_km']]))

    result = out.grab_col_names(df_final)

    cat_cols, num_cols = result[0], result[1]

    num_cols = [col for col in num_cols if
                col not in ['delivery_time', 'prepare_time', 'approved_time']]

    df_final = out.remove_all_outliers(df_final, num_cols)
    df_final.dropna(inplace=True)

    ########################################################################
    # LOGARITHMIC CONVERSION
    ########################################################################
    for i in num_cols:
        df_final[i] = np.log(df_final[i])

    ########################################################################
    # ENCODING
    ########################################################################

    df_final = en.one_hot_encoder(df_final, ['season', 'special_day', 'purchase_weekday', 'seller_state',
                                             'customer_state'], drop_first=True)

    df_final = en.label_encoder(df_final, 'cities_status')

    ########################################################################
    # FEATURE SCALING
    ########################################################################

    rs = RobustScaler()
    l = [col for col in df_final.columns if col not in ['delivery_time']]
    df_final[l] = rs.fit_transform(df_final[l])

    ########################################################################
    # MODELING
    ########################################################################

    y = df_final['delivery_time']
    X = df_final.drop(columns=['delivery_time'], axis=1)
    return X, y


########################################################################
# BASE MODEL
########################################################################
def base_models(X, y):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsRegressor()),
                   ("CART", DecisionTreeRegressor()),
                   ("RF", RandomForestRegressor()),
                   ('GBM', GradientBoostingRegressor()),
                   ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMRegressor()),
                   ('CatBoost', CatBoostRegressor(verbose=False))
                   ]
    score = pd.DataFrame(index=['rmse', 'r2_score'])
    for name, classifier in classifiers:
        rmse = np.mean(np.sqrt(-cross_val_score(classifier, X, y, cv=3, scoring="neg_mean_squared_error")))
        r2 = np.mean(cross_val_score(classifier, X, y, cv=3, scoring="r2"))
        score[name] = [rmse, r2]
        print(f'{name} hesaplandı...')
    print(score.T)


######################################################
# Automated Hyperparameter Optimization
######################################################

lightgbm_params = {"learning_rate": [0.01, 0.05, 0.1, 0.5],
                   "n_estimators": [100, 300, 500]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

classifiers = [('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMRegressor(), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=3):
    print("Hyperparameter Optimization....")
    best_models = {}
    score = pd.DataFrame(index=['rmse', 'r2_score'])
    for name, classifier, params in classifiers:
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)
        rmse = np.mean(np.sqrt(-cross_val_score(classifier, X, y, cv=3, scoring="neg_mean_squared_error")))
        r2 = np.mean(cross_val_score(classifier, X, y, cv=3, scoring="r2"))
        score[name] = [rmse, r2]
        print(f'{name} hesaplandı...')
        best_models[name] = final_model
    print(score.T)
    return best_models
