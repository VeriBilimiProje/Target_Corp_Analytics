import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Add joblib library
from datetime import datetime, timedelta
from catboost import CatBoostRegressor


# Load the pickled model

model = joblib.load('deployment/logistic_model.pkl')

# Sayfa başlığı
st.title('Delivery Time Prediction')

# Ana sayfada gösterilecek sekmeler
tab1, tab2, tab3 = st.tabs(["Tab1", "Tab2", "Tab3"])


with (tab1):
    payment_value = st.number_input("Toplam Fiyat")
    quantity = st.number_input("Adet")
    time_box= datetime(2017, 1, 1)
    time = st.date_input("Satın Alınan Tarih", time_box)
    year = time.year
    month = time.month
    days = time.weekday()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][days]
    weekday = days
    if weekday == "Monday":
        weekday = [1, 0, 0, 0, 0, 0]
    elif weekday == "Saturday":
        weekday = [0, 1, 0, 0, 0, 0]
    elif weekday == "Sunday":
        weekday = [0, 0, 1, 0, 0, 0]
    elif weekday == "Thursday":
        weekday = [0, 0, 0, 1, 0, 0]
    elif weekday == "Tuesday":
        weekday = [0, 0, 0, 0, 1, 0]
    elif weekday =="Wednesday":
        weekday = [0, 0, 0, 0, 0, 1]
    else:
        weekday = [0, 0, 0, 0, 0, 0]
    special_day = ["Normal", "Carnival", "Children", "Christmas", "New Year", "Valentine's Day", "Black Friday"]
    special_days = st.selectbox("Özel Günler", special_day)
    if special_day == "Carnival":
        special_day = [1, 0, 0, 0, 0, 0]
    elif special_day == "Children":
        special_day = [0, 1, 0, 0, 0, 0]
    elif special_day == "Christmas":
        special_day = [0, 0, 1, 0, 0, 0]
    elif special_day == "New Year":
        special_day = [0, 0, 0, 1, 0, 0]
    elif special_day == "Valentine's Day":
        special_day = [0, 0, 0, 0, 1, 0]
    elif special_day == "Normal":
        special_day = [0, 0, 0, 0, 0, 1]
    else:
        special_day = [0, 0, 0, 0, 0, 0]
with (tab2):
    c_state = ["CE" , "DF" , "ES" , "GO" , "MG" , "PE" , "PR" ,"RJ" , "RS" , "SC" , "SP", "BA" "Other"]
    customer_state = st.selectbox("Müşteri Eyaleti" , c_state)
    if c_state == "CE":
        c_state = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif c_state == "DF":
        c_state = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif c_state == "ES":
        c_state = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif c_state == "GO":
        c_state = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif c_state == "MG":
        c_state = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif c_state == "PE":
        c_state = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif c_state == "PR":
        c_state = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif c_state == "RJ":
        c_state = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif c_state == "RS":
        c_state = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif c_state == "Other":
        c_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif c_state == "SC":
        c_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif c_state == "SP":
        c_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    else:
        c_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cities_status = st.radio("Şehir Statü", ["Big", "Major", "Medium", "Small"])
    if cities_status == "Big":
        cities_status = 0
    elif cities_status == "Major":
        cities_status = 1
    elif cities_status == "Medium":
        cities_status = 2
    else:
        cities_status = 3
    state = ["PR", "RJ", "RS", "SC", "RP", "SP", "Other"]
    seller_state = ("Satıcı Eyaleti", state)
    if state == "PR":
        state = [1, 0, 0, 0, 0, 0]
    elif state == "RJ":
        state = [0, 1, 0, 0, 0, 0]
    elif state == "RS":
        state = [0, 0, 1, 0, 0, 0]
    elif state == "Other":
        state = [0, 0, 0, 1, 0, 0]
    elif state == "SC":
        state = [0, 0, 0, 0, 1, 0]
    elif state == "SP":
        state = [0, 0, 0, 0, 0, 1]
    else:
        state = [0, 0, 0, 0, 0, 0]
    prepare_time = st.number_input("Ürünün Hazırlanma Süresi" , step=1)

with (tab3):
    product_weight_g = st.number_input("Ürün Ağırlığı", min_value=1)
    product_cm3 = st.number_input("Ürün cm3'ü")
    cargo_score = st.selectbox("Kargo Şirketi", ["A Sınıfı", "B Sınıfı", "C Sınıfı", "D Sınıfı", "E Sınıfı"])
    if cargo_score == "E Sınıfı":
        cargo_score = 25
    elif cargo_score == "D Sınıfı":
        cargo_score = 60
    elif cargo_score == "C Sınıfı":
        cargo_score = 100
    elif cargo_score == "B Sınıfı":
        cargo_score = 220
    else:
        cargo_score = 500

    distance_km = st.slider('Mesafe', min_value=1, max_value=8736, value=1)
    season = ""
    if month < 3:
        season = "q1"
    elif month < 6:
        season = "q2"
    elif month < 9:
        season = "q3"
    else:
        season = "q4"

    if season == "q1":
        season = [0,0,0]
    elif season == "q2":
        season = [1,0,0]
    elif season == "q3":
        season = [0,1,0]
    else:
        season = [0,0,1]

    def predict_delivery_time(weight_g_p, payment_v_p, distance_p, cities_p, quantity_p, year_p, cm3_p,
                             month_p, prepare_p, cargo_p, season_p_0,season_p_1,season_p_2, special_p_0,special_p_1,special_p_2,special_p_3,special_p_4,special_p_5,
                              weekday_p_0,weekday_p_1,weekday_p_2,weekday_p_3,weekday_p_4,weekday_p_5,seller_s_p_0,seller_s_p_1,seller_s_p_2,seller_s_p_3,seller_s_p_4,seller_s_p_5,
                              customer_s_p_0,customer_s_p_1,customer_s_p_2,customer_s_p_3,customer_s_p_4,customer_s_p_5,customer_s_p_6,customer_s_p_7,customer_s_p_8,customer_s_p_9,
                              customer_s_p_10,customer_s_p_11):

        features = [weight_g_p, payment_v_p, distance_p, cities_p, quantity_p, year_p, cm3_p,
                             month_p, prepare_p, cargo_p, season_p_0,season_p_1,season_p_2, special_p_0,special_p_1,special_p_2,special_p_3,special_p_4,special_p_5,
                              weekday_p_0,weekday_p_1,weekday_p_2,weekday_p_3,weekday_p_4,weekday_p_5,seller_s_p_0,seller_s_p_1,seller_s_p_2,seller_s_p_3,seller_s_p_4,seller_s_p_5,
                              customer_s_p_0,customer_s_p_1,customer_s_p_2,customer_s_p_3,customer_s_p_4,customer_s_p_5,customer_s_p_6,customer_s_p_7,customer_s_p_8,customer_s_p_9,
                              customer_s_p_10,customer_s_p_11]
        # Make the prediction using the loaded model
        prediction = model.predict([features])

        # Return the predicted review score
        return prediction

    if st.button('Tahmin Et'):
        # Call the prediction function with input features
        predicted_score = predict_delivery_time(product_weight_g, payment_value, distance_km, cities_status,
                                               quantity, year, product_cm3, month, prepare_time, cargo_score, *season,
                                               *special_day, *weekday, *state, *c_state )

        st.write('Tahmin Edilen Delivery Time:',time + timedelta(int(predicted_score)))



