import pandas as pd
from lib import outliers as out, summary as sum, graphic as gra

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)

df = pd.read_csv('datasets/dataset.csv')

df.shape

df.nunique()

df.head()

result = out.grab_col_names(df)

cat_cols, num_cols = result[0], result[1]

gra.plot_numerical_col(df, num_cols=num_cols, plot_type='kde')

gra.plot_categoric_col(df, cat_cols)

sum.check_df(df)

sum.cat_summary(df, cat_cols)

for col in num_cols:
    sum.target_summary_with_num(df, 'review_score', numerical_col=col)

for col in cat_cols:
    sum.target_summary_with_cat(df, 'review_score', categorical_col=col)

sum.rare_analyser(df, 'review_score', cat_cols)

sum.correlation_matrix(df, num_cols)

sum.missing_values_table(df, num_cols)

out.for_check(df, num_cols)
# # 4) Hangi ayda en fazla sipariş verilmektedir?
# #4.1 bu şekilde ayın numarasını verir
# #df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
# #en_fazla_siparis_verilen_ay = df['order_purchase_timestamp'].dt.month.value_counts().idxmax()
# #print("En fazla sipariş verilen ay:", en_fazla_siparis_verilen_ay)
# #4.2 ayın ismini istersek
# #import calendar
# 
# # En fazla sipariş verilen ayın numarasını al
# #en_fazla_siparis_verilen_ay_num = df['order_purchase_timestamp'].dt.month.value_counts().idxmax()
# 
# # Ayın adını alın
# #en_fazla_siparis_verilen_ay_ad = calendar.month_name[en_fazla_siparis_verilen_ay_num]
# 
# # Sonuc
# print("En fazla sipariş verilen ay:", en_fazla_siparis_verilen_ay_ad)
# #4.3 Grafik ile görmek istersek
# 
# # Siparişlerin alındığı ayın çıkarılması
# df['order_month'] = df['order_purchase_timestamp'].dt.month
# 
# # Her ayın sipariş sayısını hesaplanması
# siparis_aylara_gore = df['order_month'].value_counts().sort_index()
# 
# # Çubuk grafiği oluşturma
# plt.figure(figsize=(10, 6))
# siparis_aylara_gore.plot(kind='bar', color='skyblue')
# 
# # Grafik başlığını ve eksen etiketlerini eklenmesi
# # plt.title('Hangi Ayda En Fazla Sipariş Verilmektedir?')
# #plt.xlabel('Ay')
# #plt.ylabel('Sipariş Sayısı')
# 
# # Grafikleri göster
#  plt.show()
