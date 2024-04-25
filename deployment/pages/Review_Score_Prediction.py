import streamlit as st
import pandas as pd
import numpy as np

# Sayfa başlığı
st.title('Review Score Prediction')

# Ana sayfada gösterilecek sekmeler
tab1, tab2, tab3 = st.tabs(["Ücretler", "Ödeme Yöntemi ve Kategori", "Süreç"])

with tab1:
   price = st.number_input('Ürün Fiyatı', min_value=0.0)
   freight_value = st.number_input('Kargo Ücreti', min_value=0.0)
   discount = st.number_input('İndirim Oranı', min_value=0.0)

with tab2:
   payment_type = st.radio("Ödeme Yöntemi", ["Debit Card", "Credit Card", "Coupon"])
   payment_installments = st.slider('Ödeme Taksit Sayısı', min_value=1, max_value=24, value=1)
   categories = ['Furniture', 'Automotive', 'Sport & Leisure', 'Home & Garden', 'Baby & Kids',
                 'Electronics', 'Fashion & Accessories', 'Food & Beverages', 'Books & Media',
                 'Office & Stationary', 'Beauty & Health', 'Construction & Tools', 'Pets',
                 'Electricals', 'Media & Entertainment', 'Industry & Business',
                 'Insurance & Services', 'Flowers & Decorations']
   category = st.selectbox('Ürün Kategorisi', categories)

with tab3:
   approval_time = st.number_input('Onay Süresi (dk)', min_value=0)
   customer_wait_time = st.number_input('Müşteri Bekleme Süresi (gün)', min_value=0)
   delivery_time_diff = st.number_input('Teslimat Süresi Farkı (gün)', min_value=0)
   days_to_delivery_actual = st.number_input('Gerçek Teslimat Gün Sayısı', min_value=0)
   seller_review_score = st.number_input('Satıcı Değerlendirme Puanı', min_value=0.0, max_value=5.0, step=0.1)
   if st.button('Tahmin Et'):
       # Burada girdileri kullanarak tahmini yapabilirsiniz
       # Örneğin, bir tahmin fonksiyonu çağırabilir ve sonucu ekrana yazdırabilirsiniz
       # Tahmin edilen puanı prediction değişkenine atayabilir ve alt kısımda yazdırabilirsiniz
       prediction = 4  # Örnek bir tahmin
       st.write('Tahmin Edilen Review Score:', prediction)

# Ana sayfanın genişliğini ayarla (%50)
st.markdown("""
    <style>
        .reportview-container .main .block-container {
            max-width: 50%;
            padding-right: 25%;
            padding-left: 25%;
            display: flex;
            justify-content: space-between;
        }
    </style>
""", unsafe_allow_html=True)

