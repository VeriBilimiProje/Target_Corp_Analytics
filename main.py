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

df = pd.merge(df, dff, on='seller_id')

df.drop(['customer_unique_id', 'customer_zip_code_prefix', 'payment_sequential', 'review_comment_title',
         'product_name_length',
         'product_description_length', 'seller_zip_code_prefix', 'product_photos_qty'
         ], inplace=True, axis=1)

df.dropna(subset=['product category', 'product_weight_g', 'product_length_cm'
    , 'product_height_cm', 'product_width_cm', 'order_approved_at', 'order_delivered_carrier_date',
                  'order_delivered_customer_date'], inplace=True)

df = df.drop_duplicates(subset=['order_id'])

gra.plot_numerical_col(df, num_cols=['price', 'payment_value', 'freight_value'], plot_type='kde')

df = out.replace_all_outliers(df, ['price', 'payment_value', 'freight_value', 'product_weight_g', 'product_width_cm'])

out.for_check(df, df.columns)

result = out.grab_col_names(df)

cat_cols, num_cols = result[0], result[1]

df['product category'].unique()

categories_dict = {
    'Furniture': ['Furniture office', 'Furniture Decoration', 'Room Furniture',
                  'Furniture Kitchen Service Area Dinner and Garden', 'Furniture',
                  'CITTE AND UPHACK FURNITURE', 'bed table bath', ],

    'Automotive': ['automotive'],

    'Sport & Leisure': ['sport leisure', 'Toys', 'Games consoles',
                        'Fashion Sport'],

    'Home & Garden': ['House comfort', 'House Comfort 2', 'Cool Stuff',
                      'Garden tools', 'Christmas articles', 'party articles',
                      'Construction Tools Garden', 'HOUSE PASTALS OVEN AND CAFE',
                      'La Cuisine', 'housewares', 'climatization', 'Market Place'],

    'Baby & Kids': ['babies', 'toys', 'Fashion Children\'s Clothing'],

    'Electronics': ['computer accessories', 'musical instruments', 'Electronics',
                    'ELECTRICES 2', 'PCs', 'IMAGE IMPORT TABLETS',
                    'home appliances', 'Kitchen portable and food coach',
                    'PCs', 'telephony', 'electronics', 'fixed telephony', 'PC Gamer'],

    'Fashion & Accessories': ['Fashion Bags and Accessories',
                              'Fashion Women\'s Clothing',
                              'Fashion Men\'s Clothing',
                              'Fashion Underwear and Beach Fashion', 'Watches present', 'Bags Accessories',
                              'Fashion Calcados'],

    'Food & Beverages': ['Drink foods', 'drinks', 'foods'],

    'Books & Media': ['General Interest Books', 'technical books',
                      'Imported books', 'Arts and Crafts', 'Art', 'audio'],

    'Office & Stationary': ['stationary store'],

    'Beauty & Health': ['HEALTH BEAUTY', 'Hygiene diapers', 'perfumery'],

    'Construction & Tools': ['Casa Construcao',
                             'Construction Tools Construction',
                             'Construction Tools Illumination',
                             'Construction Tools Garden',
                             'Construction Tools Tools',
                             'CONSTRUCTION SECURITY TOOLS'],

    'Pets': ['pet Shop'],

    'Electricals': ['electrostile',
                    'SIGNALIZATION AND SAFETY'],

    'Media & Entertainment': ['song', 'Blu Ray DVDs', 'cine photo',
                              'cds music dvds'],

    'Industry & Business': ['Agro Industria e Comercio',
                            'Industry Commerce and Business'],

    'Insurance & Services': ['insurance and services'],

    'Flowers & Decorations': ['flowers']
}

print('##################################')
df['category'] = ''

# Her bir kategori için döngü
for v in df['product category'].unique():
    for key, value_list in categories_dict.items():
        if v in value_list:
            df.loc[df['product category'] == v, 'category'] = key

df.nunique()

df.tail(20)

df.isnull().sum()

df.drop('product category', axis=1, inplace=True)
