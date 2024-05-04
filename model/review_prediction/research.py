import numpy as np
import pandas as pd
from lib import outliers as out, encoding as en
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)


def data_prep(df):
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

    result = out.grab_col_names(df)
    cat_cols, num_cols = result[0], result[1]

    out.replace_all_outliers(df, num_cols)

    seller = df.seller_id.value_counts().to_dict()
    seller_popularity = []
    for _id in df.seller_id.values:
        seller_popularity.append(seller[_id])
    df['seller_popularity'] = seller_popularity

    df['seller_popularity'] = pd.cut(df['seller_popularity'], bins=[0, 300, 1000, np.inf], labels=['C', 'B', 'A'])

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

    df['season'] = df['order_purchase_timestamp'].dt.month

    label = ['q1', 'q2', 'q3', 'q4']

    df['season'] = pd.qcut(df['season'], 4, label)

    df['year'] = df['order_purchase_timestamp'].dt.year

    seller_review_score = df.groupby('seller_id')['review_score'].mean()

    df['seller_review_score'] = df['seller_id'].map(seller_review_score * 2)

    df['customer_wait_day'] = pd.cut(df['customer_wait_day'], bins=[0, 8, 16, 25, 40, 61],
                                     labels=['Very_Fast', 'Fast', 'Neutral', 'Slow', 'Worst'])

    df['estimated_days'] = pd.cut(df['estimated_days'], bins=[0, 8, 16, 25, 40, 61],
                                  labels=['Very_Fast', 'Fast', 'Neutral', 'Slow', 'Worst'])

    df['ships_in'] = pd.cut(df['ships_in'], bins=[0, 4, 8, 16, 28, 61],
                            labels=['Very_Fast', 'Fast', 'Neutral', 'Slow', 'Worst'])

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

    df.drop(df[df['review_score'] == 3].index, inplace=True)

    df[['payment_type']] = en.rare_encoder(df[['payment_type']], 0.06)
    df[['product category']] = en.rare_encoder(df[['product category']], 0.02)

    encoded_class = {1: 'Not Satisfied',
                     2: 'Not Satisfied',
                     3: 'Not Satisfied',
                     4: 'Satisfied',
                     5: 'Satisfied'}

    df['review_score'] = df['review_score'].map(encoded_class)

    df_final = df[
        ['review_score', 'price', 'freight_value', 'payment_type', 'payment_installments', 'payment_value',
         'customer_wait_day', 'seller_review_score', 'delay_time', 'distance_km', 'seller_popularity', 'discount'
         ]]

    df_final['price'] = np.log(df_final['price'])
    df_final['payment_value'] = np.log(df_final['payment_value'])
    df_final['seller_review_score'] = np.log(df_final['seller_review_score'])
    df_final['distance_km'] = np.log(df_final['distance_km'])

    df_final['delay_time'].fillna(-14, inplace=True)
    df_final['distance_km'].fillna(428, inplace=True)
    df_final['customer_wait_day'].fillna('Worst', inplace=True)
    df_final.dropna(inplace=True)

    df_final = en.one_hot_encoder(df_final, ['payment_type', 'customer_wait_day', 'seller_popularity'], drop_first=True)

    df_final = en.label_encoder(df_final, 'review_score')

    rs = RobustScaler()
    l = [col for col in df_final.columns if col not in ['review_score']]
    df_final[l] = rs.fit_transform(df_final[l])

    y = df_final['review_score']
    X = df_final.drop(columns=['review_score'], axis=1)

    return X, y


def base_models(X, y):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]
    score = pd.DataFrame(index=['accuracy', 'f1', 'roc_auc'])
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
        f1 = cv_results['test_f1'].mean()
        auc = cv_results['test_roc_auc'].mean()
        accuracy = cv_results['test_accuracy'].mean()
        score[name] = [accuracy, f1, auc]
        print(f'{name} hesaplandı...')
    print(score.T)


######################################################
# 4. Automated Hyperparameter Optimization
######################################################

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    score = pd.DataFrame(index=['accuracy', 'f1', 'roc_auc'])
    for name, classifier, params in classifiers:
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'])
        f1 = cv_results['test_f1'].mean()
        auc = cv_results['test_roc_auc'].mean()
        accuracy = cv_results['test_accuracy'].mean()
        score[name] = [accuracy, f1, auc]
        print(f'{name} hesaplandı...')
        best_models[name] = final_model
    print(score.T)
    return best_models
