import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Add joblib library

# Load the pickled model
model = joblib.load('deployment/log_reg.pkl')  # Replace 'log_reg.pkl' with your actual filename

# Sayfa başlığı
st.title('Review Score Prediction')

# Ana sayfada gösterilecek sekmeler
tab1, tab2, tab3 = st.tabs(["Ücretler", "Ödeme Yöntemi ve Kategori", "Süreç"])

with tab1:
    price = st.number_input('Ürün Fiyatı', min_value=0.0)
    freight_value = st.number_input('Kargo Ücreti', min_value=0.0)
    discount = st.number_input('İndirim Oranı', min_value=0.0)

    # İleri butonu (Tab2'yi açar) - Unique key added
    if st.button('→ İleri', key='forward_tab1'):
        st.session_state.current_tab = "Ödeme Yöntemi ve Kategori"

with tab2:
    payment_type = st.radio("Ödeme Yöntemi", ["Debit Card", "Credit Card", "Coupon"])
    payment_installments = st.slider('Ödeme Taksit Sayısı', min_value=1, max_value=24, value=1)
    categories = ['Furniture', 'Automotive', 'Sport & Leisure', 'Home & Garden', 'Baby & Kids',
                  'Electronics', 'Fashion & Accessories', 'Food & Beverages', 'Books & Media',
                  'Office & Stationary', 'Beauty & Health', 'Construction & Tools', 'Pets',
                  'Electricals', 'Media & Entertainment', 'Industry & Business',
                  'Insurance & Services', 'Flowers & Decorations']
    category = st.selectbox('Ürün Kategorisi', categories)

    # İleri butonu (Tab3'ü açar) - Unique key added
    if st.button('→ İleri', key='forward_tab2'):
        st.session_state.current_tab = "Süreç"

with tab3:
    approval_time = st.number_input('Onay Süresi (dk)', min_value=0)
    customer_wait_time = st.number_input('Müşteri Bekleme Süresi (gün)', min_value=0)
    delivery_time_diff = st.number_input('Teslimat Süresi Farkı (gün)', min_value=0)
    days_to_delivery_actual = st.number_input('Gerçek Teslimat Gün Sayısı', min_value=0)
    seller_review_score = st.number_input('Satıcı Değerlendirme Puanı', min_value=0.0, max_value=5.0, step=0.1)

    # Define the prediction function
    def predict_review_score(price, freight_value, discount, payment_type, payment_installments,
                             category, approval_time, customer_wait_time, delivery_time_diff,
                             days_to_delivery_actual, seller_review_score):
        # Prepare the input features
        features = [price, freight_value, discount, payment_type, payment_installments,
                   category, approval_time, customer_wait_time, delivery_time_diff,
                   days_to_delivery_actual, seller_review_score]

        # Make the prediction using the loaded model
        prediction = model.predict([features])

        # Return the predicted review score
        return prediction[0]

    # Predict the review score
    if st.button('Tahmin Et'):
        # Gather input features from all tabs
        input_features = [price, freight_value, discount, payment_type, payment_installments,
                         category, approval_time, customer_wait_time, delivery_time_diff,
                         days_to_delivery_actual, seller_review_score]

        # Call the prediction function
        predicted_score = predict_review_score(*input_features)  # Unpack the list

        # Display the predicted review score
        st.write('Tahmin Edilen Review Score:', predicted_score)
