import streamlit as st
import pandas as pd
import numpy as np
import time

df = pd.read_csv('datasets/final_dataset.csv')

# Sayfa başlığı
st.title('Review Score Prediction')

# Girdi bilgileri formu
st.write('Lütfen aşağıdaki bilgileri doldurun:')

# Fiyat girişi
price = st.number_input('Ürün Fiyatı', min_value=0.0)

# Kargo ücreti girişi
freight_value = st.number_input('Kargo Ücreti📦', min_value=0.0)

# Ödeme türü seçimi
payment_value = st.radio(
        "Ödeme Yöntemi👇",
        ["Debit Card💵", "Credit Card💳", "Coupon🔖"],
        key="Credit Card",)
# Ödeme taksitleri girişi
payment_installments = st.slider('Ödeme Taksit Sayısı', min_value=1, max_value=24, value=1)

# Ödeme tutarı girişi
payment_value = st.number_input('Ödeme Tutarı', min_value=0.0)

# Ürün kategorisi seçimi
categories = ['Furniture', 'Automotive', 'Sport & Leisure', 'Home & Garden', 'Baby & Kids',
              'Electronics', 'Fashion & Accessories', 'Food & Beverages', 'Books & Media',
              'Office & Stationary', 'Beauty & Health', 'Construction & Tools', 'Pets',
              'Electricals', 'Media & Entertainment', 'Industry & Business',
              'Insurance & Services', 'Flowers & Decorations']
categories.sort()  # Kategorileri alfabetik olarak sırala
category = st.selectbox('Ürün Kategorisi', categories)

# Onay süresi girişi
approval_time = st.number_input('Onay Süresi (dk)', min_value=0)

# Müşteri bekleme süresi girişi
customer_wait_time = st.number_input('Müşteri Bekleme Süresi (gün)', min_value=0)

# İndirim girişi
discount = st.number_input('İndirim Oranı', min_value=0.0)

# Teslimat süresi farkı girişi
delivery_time_diff = st.number_input('Teslimat Süresi Farkı (gün)', min_value=0)

# Fiyat/kargo oranı girişi
price_freight_ratio = st.number_input('Fiyat/Kargo Oranı')

# Gerçek teslimat gün sayısı girişi
days_to_delivery_actual = st.number_input('Gerçek Teslimat Gün Sayısı', min_value=0)

# Satıcı değerlendirme puanı girişi
seller_review_score = st.number_input('Satıcı Değerlendirme Puanı', min_value=0.0, max_value=5.0, step=0.1)

# Tahmin düğmesi
if st.button('Tahmin Et'):
    # Burada girdileri kullanarak tahmini yapabilirsiniz
    # Örneğin, bir tahmin fonksiyonu çağırabilir ve sonucu ekrana yazdırabilirsiniz
    # Tahmin edilen puanı prediction değişkenine atayabilir ve alt kısımda yazdırabilirsiniz
    prediction = 4  # Örnek bir tahmin
    st.write('Tahmin Edilen Review Score:', prediction)

