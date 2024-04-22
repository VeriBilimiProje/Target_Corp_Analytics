import numpy as np
import pandas as pd
from lib import encoding as en, outliers as out, summary as sum, graphic as gra
from lib import outliers as out, summary as sum, encoding as en
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)

customers_df = pd.read_csv('datasets/olist_customers_dataset.csv')
geolocation_df = pd.read_csv('datasets/olist_geolocation_dataset.csv')
items_df = pd.read_csv('datasets/olist_order_items_dataset.csv')
payments_df = pd.read_csv('datasets/olist_order_payments_dataset.csv')
reviews_df = pd.read_csv('datasets/olist_order_reviews_dataset.csv')
orders_df = pd.read_csv('datasets/olist_orders_dataset.csv')
products_df = pd.read_csv('datasets/olist_products_dataset.csv')
sellers_df = pd.read_csv('datasets/olist_sellers_dataset.csv')
category_translation_df = pd.read_csv('datasets/product_category_name_translation.csv')

df = pd.merge(customers_df, orders_df, on="customer_id", how='inner')
df = df.merge(reviews_df, on="order_id", how='inner')
df = df.merge(items_df, on="order_id", how='inner')
df = df.merge(products_df, on="product_id", how='inner')
df = df.merge(payments_df, on="order_id", how='inner')
df = df.merge(sellers_df, on='seller_id', how='inner')
df = df.merge(category_translation_df, on='product_category_name', how='inner')

df.rename(columns={'product_category_name_english': 'product category', 'product_name_lenght': 'product_name_length',
                   'product_description_lenght': 'product_description_length'}, inplace=True)

df.drop(['customer_unique_id', 'customer_zip_code_prefix', 'payment_sequential', 'review_comment_title',
         'product_name_length',
         'product_description_length', 'seller_zip_code_prefix', 'product_photos_qty', 'product_category_name'
            , 'review_comment_message'
         ], inplace=True, axis=1)

df.dropna(subset=['product category', 'product_weight_g', 'product_length_cm'
    , 'product_height_cm', 'product_width_cm', 'order_approved_at', 'order_delivered_carrier_date',
                  'order_delivered_customer_date'], inplace=True)

df = out.remove_all_outliers(df, ['price', 'payment_value', 'freight_value', 'product_weight_g', 'product_width_cm'])

result = out.grab_col_names(df)

cat_cols, num_cols = result[0], result[1]

gra.plot_numerical_col(df, num_cols=['payment_value'], plot_type='kde')

categories_dict = {
    'Furniture': ['office_furniture', 'furniture_decor', 'furniture_living_room',
                  'kitchen_dining_laundry_garden_furniture', 'furniture_bedroom',
                  'furniture_mattress_and_upholstery', 'bed_bath_table', ],

    'Automotive': ['auto'],

    'Sport & Leisure': ['sports_leisure', 'toys', 'consoles_games',
                        'fashion_sport'],

    'Home & Garden': ['home_confort', 'home_comfort_2', 'cool_stuff',
                      'garden_tools', 'christmas_supplies', 'party_supplies', 'home_construction',
                      'costruction_tools_garden', 'small_appliances_home_oven_and_coffee', 'small_appliances',
                      'la_cuisine', 'housewares', 'air_conditioning', 'market_place'],

    'Baby & Kids': ['baby', 'toys', 'fashion_childrens_clothes'],

    'Electronics': ['computers_accessories', 'musical_instruments', 'Electronics',
                    'home_appliances_2', 'computers', 'tablets_printing_image',
                    'home_appliances', 'Kitchen portable and food coach',
                    'PCs', 'telephony', 'electronics', 'fixed_telephony', 'PC Gamer'],

    'Fashion & Accessories': ['fashion_bags_accessories',
                              'fashio_female_clothing',
                              'fashion_male_clothing',
                              'fashion_underwear_beach', 'watches_gifts', 'luggage_accessories',
                              'fashion_shoes'],

    'Food & Beverages': ['food_drink', 'drinks', 'food'],

    'Books & Media': ['books_general_interest', 'books_technical',
                      'books_imported', 'arts_and_craftmanship', 'art', 'audio'],

    'Office & Stationary': ['stationery'],

    'Beauty & Health': ['health_beauty', 'diapers_and_hygiene', 'perfumery'],

    'Construction & Tools': ['construction_tools_lights',
                             'construction_tools_construction',
                             'Construction Tools Illumination',
                             'Construction Tools Garden',
                             'costruction_tools_tools',
                             'construction_tools_safety'],

    'Pets': ['pet_shop'],

    'Electricals': ['electronics',
                    'signaling_and_security'],

    'Media & Entertainment': ['music', 'dvds_blu_ray', 'cine_photo',
                              'cds_dvds_musicals'],

    'Industry & Business': ['industry_commerce_and_business',
                            'agro_industry_and_commerce'],

    'Insurance & Services': ['security_and_services'],

    'Flowers & Decorations': ['flowers']
}

df['category'] = ''

for v in df['product category'].unique():
    for key, value_list in categories_dict.items():
        if v in value_list:
            df.loc[df['product category'] == v, 'category'] = key

df.drop('product category', axis=1, inplace=True)

list = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date', 'review_creation_date', 'review_answer_timestamp',
        'shipping_limit_date']

df['review_creation_date'] = df['review_creation_date'].astype('datetime64[ns]')
df['review_answer_timestamp'] = df['review_answer_timestamp'].astype('datetime64[ns]')

for col in list:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')

corr = df[num_cols].corr()
cor_matrix = corr.abs()
upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix if any(upper_triangle_matrix[col] > 0.9)]
# df.drop(drop_list, inplace=True, axis=1)
sns.set(rc={'figure.figsize': (8, 8)})
sns.heatmap(corr, cmap='RdBu', annot=True)
plt.show()

df['approval_time(dk)'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / 60

df['customer_wait_time(day)'] = (df['order_delivered_customer_date'] - df[
    'order_purchase_timestamp']).dt.total_seconds() / 86400

df['max_price'] = df.groupby('product_id')['price'].transform('max')
df['discount'] = 100 - ((df['price'] / df['max_price']) * 100)
df.drop(columns=['max_price'], inplace=True)

df['purchase_weekday'] = df['order_purchase_timestamp'].dt.weekday

df.head()

df['purchase_weekday'] = df['purchase_weekday'].replace({5: 0, 6: 0, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1})

special_days = {
    'New Year': ['01-01'],
    'Carnival': ['02-24', '02-25', '02-26'],
    'Valentine\'s Day': ['06-12'],
    'Children\'s Day': ['10-12'],
    'Black Friday': ['11-27'],
    'Christmas': ['12-25']
}

df['special_day'] = 0

for event, dates in special_days.items():
    for date in dates:
        df.loc[df['order_purchase_timestamp'].dt.strftime('%m-%d') == date, 'special_day'] = 1

df['delivery_time_diff'] = (df['order_estimated_delivery_date'] - df[
    'order_delivered_customer_date']).dt.total_seconds() / 86400

df['price_freight_ratio'] = df['freight_value'] / df['price']

df['product_volume_m3'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm'] / 100

df['days_to_delivery_actual'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

df.drop(df[df['order_status'] == 'canceled'].index, inplace=True)

df.drop(['order_status'], axis=1, inplace=True)

df['season'] = df['order_purchase_timestamp'].dt.month

label = ['q1', 'q2', 'q3', 'q4']

df['season'] = pd.qcut(df['season'], 4, label)

########################################################################
df['year'] = df['order_purchase_timestamp'].dt.year

df.head()


df['product_id'].value_counts()

earnings_by_year_and_season = df.groupby(['year', 'season'])['payment_value'].sum()

total_earnings = df.apply(lambda row: earnings_by_year_and_season[(row['year'], row['season'])], axis=1)

df['total_earn_quantile'] = total_earnings

earnings_by_year_season_seller = df.groupby(['year', 'season', 'seller_id'])['payment_value'].sum()

total_earning_by_seller = df.apply(
    lambda row: earnings_by_year_season_seller.get((row['year'], row['season'], row['seller_id']), 0), axis=1)

df['total_earn_quantile_by_seller'] = total_earning_by_seller

df.groupby(['season', 'year'])['total_earn_quantile'].mean()

seller_review_score = df.groupby('seller_id')['review_score'].mean()

df['seller_review_score'] = df['seller_id'].map(seller_review_score * 2)

df['seller_id'].value_counts()


df[df['freight_value'] > 50]['review_score'].mean()

df['product_id'].value_counts()

df['review_score'].value_counts()

df.head()

result = out.grab_col_names(df)

cat_cols, num_cols = result[0], result[1]

sum.cat_summary(df,cat_cols)

for col in num_cols:
    sum.target_summary_with_num(df, 'review_score', col)
########################################################################

df.groupby('category')['review_score'].mean()

df_glad = df[
    ['review_score', 'price', 'freight_value', 'payment_type', 'payment_installments', 'payment_value', 'category',
     'approval_time(dk)', 'customer_wait_time(day)',
     'discount', 'delivery_time_diff', 'price_freight_ratio', 'days_to_delivery_actual', 'seller_review_score'
     ]]

rs = RobustScaler()
l = [col for col in df_glad.columns if col not in ['review_score', 'payment_type', 'category']]
df_glad[l] = rs.fit_transform(df_glad[l])

df_glad = en.one_hot_encoder(df_glad, ['payment_type', 'category'], drop_first=True)

df_glad.head()

y = df_glad['review_score']
X = df_glad.drop(columns=['review_score'], axis=1)

l_model = LinearRegression().fit(X, y)
y_pred = l_model.predict(X)

np.sqrt(mean_squared_error(y, y_pred))

df_glad['review_score'].mean()

r2_score(y, y_pred)

from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

forest_reg = RandomForestRegressor(random_state=42)

forest_reg.get_params()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [1, 3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(forest_reg, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

params = {'max_depth': None,
          'max_features': 7,
          'min_samples_split': 20,
          'n_estimators': 500}

forest_reg = forest_reg.set_params(**params)

forest_reg.fit(X_train, y_train)

y_pred = forest_reg.predict(X_train)
forest_mse = mean_squared_error(y_train, y_pred)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

cv_results = cross_validate(forest_reg, X_train, y_train, cv=5)
print("Ortalama eğitim doğruluğu:", cv_results['train_score'].mean())

pd.DataFrame({'y': y_train, 'y_pred': y_pred})

