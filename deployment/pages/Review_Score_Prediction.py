import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Add joblib library
from datetime import datetime, timedelta
import os
import logging
from streamlit_extras.let_it_rain import rain

# Load the model
model = joblib.load("deployment/review_model.pkl")

# Sayfa başlığı
col1, col2, col3 = st.columns(3)

# İlk metrik
with col1:
    container1 = st.container(border=True)
    rmse = container1.metric(label="Accuracy" , value="0.881", delta="0.04")

# İkinci metrik
with col2:
    container2 = st.container(border=True)
    r2 = container2.metric(label="F1 Score" , value="0.933", delta="0.02")

with col3:
    container3 = st.container(border=True)
    r2 = container3.metric(label="Roc Auc" , value="0.794", delta="0.228")
st.title('Review Score Prediction')

# Ana sayfada gösterilecek sekmeler
tab1, tab2, tab3 = st.tabs(["Ücretler", "Ödeme Yöntemi ve Kategori", "Süreç"])

with tab1:
    price = st.number_input('Ürün Fiyatı', min_value=0.0)
    freight_value = st.number_input('Kargo Ücreti', min_value=0.0)
    discount = st.number_input('İndirim Oranı', min_value=0.0)
    quantity = st.number_input('Ürün Adedi', min_value=1 , max_value=5)

    # İleri butonu (Tab2'yi açar) - Unique key added
    if st.button('→ İleri', key='forward_tab1'):
        st.session_state.current_tab = "Ödeme Yöntemi ve Kategori"

with tab2:
    payment_type = st.radio("Ödeme Yöntemi", ["Debit Card", "Credit Card", "Coupon"])
    payment_installments = st.slider('Ödeme Taksit Sayısı', min_value=1, max_value=24, value=1)
    categories = ['Onaylanmış Satıcı', 'Başarılı Satıcı', 'Onaylanmamış Satıcı']
    category = st.selectbox('Satıcı Tipi', categories)

    if category == "Onaylanmış Satıcı":
        category = [1, 0]
    elif category == "Başarılı Satıcı":
        category = [0, 1]
    else:
        category = [0, 0]

    if payment_type == "Debit Card":
        payment_type = [1, 0]
    elif payment_type == "Credit Card":
        payment_type = [0, 1]
    else:
        payment_type = [0, 0]

    # İleri butonu (Tab3'ü açar) - Unique key added
    if st.button('→ İleri', key='forward_tab2'):
        st.session_state.current_tab = "Süreç"

with tab3:
    default_purchase_date = datetime(2017, 1, 1)
    order_purchase_timestamp = st.date_input('Satın Alınan Tarih', value=default_purchase_date)

    # order_delivered_carrier_date için varsayılan başlangıç tarihi (order_purchase_timestamp'tan bir gün sonrası)
    if order_purchase_timestamp:
        default_carrier_date = order_purchase_timestamp + timedelta(days=1)
    else:
        default_carrier_date = default_purchase_date + timedelta(days=1)
    order_delivered_estimated_date = st.date_input('Ulaşması Gereken Tarih', value=default_carrier_date)

    # order_delivered_timestamp için varsayılan başlangıç tarihi (order_delivered_carrier_date'ten bir gün sonrası)
    if order_delivered_estimated_date:
        default_delivery_date = order_delivered_estimated_date + timedelta(days=1)
    else:
        default_delivery_date = default_purchase_date + timedelta(days=1)
    order_delivered_timestamp = st.date_input('Müşteriye Ulaşan Tarih', value=default_delivery_date)

    seller_review_score = st.number_input('Satıcı Değerlendirme Puanı', min_value=0, max_value=10, step=1)
    distance_km = st.slider('Mesafe', min_value=1, max_value=8736, value=1)
    customer_wait_day = (order_delivered_timestamp - order_purchase_timestamp).total_seconds() / 86400
    payment_value = ((price + freight_value) * quantity) - discount
    delay_time = (order_delivered_estimated_date - order_delivered_timestamp).total_seconds() / 86400

    if customer_wait_day <= 8:
        customer_wait_day = [0, 0, 0, 0]
    elif customer_wait_day <= 16:
        customer_wait_day = [1, 0, 0, 0]
    elif customer_wait_day <= 25:
        customer_wait_day = [0, 1, 0, 0]
    elif customer_wait_day <= 40:
        customer_wait_day = [0, 0, 1, 0]
    else:
        customer_wait_day = [0, 0, 0, 1]


    def predict_review_score(price_p, freight_value_p, installments_p, value_p, seller_score_p, delay_p, distance_p,
                             discount_p, payment_type_0, payment_type_1, wait_p_0, wait_p_1, wait_p_2, wait_p_3,
                             popularity_p_0, popularity_p_1):

        features = [price_p, freight_value_p, installments_p, value_p, seller_score_p, delay_p, distance_p, discount_p,
                    payment_type_0, payment_type_1, wait_p_0, wait_p_1, wait_p_2, wait_p_3, popularity_p_0,
                    popularity_p_1]
        # Make the prediction using the loaded model
        prediction = model.predict([features])

        # Return the predicted review score
        return prediction[0]


    # Predict the review score
    if st.button('Tahmin Et'):
        # Call the prediction function with input features
        predicted_score = predict_review_score(price, freight_value, payment_installments, payment_value,
                                               seller_review_score, delay_time, distance_km, discount, payment_type[0],
                                               payment_type[1], *customer_wait_day, *category)


    def example():
        rain(
            emoji="😡",
            font_size=100,
            falling_speed=3,
            animation_length="1")
    try:
        if int(predicted_score) == 0:
            st.error("😡 Unsatisfied")
            example()
        else:
            st.balloons()
            st.success("🤩 Satisfied")
    except:
        st.write("")
