import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Add joblib library
from datetime import datetime, timedelta
from catboost import CatBoostRegressor


# Load the pickled model

model = joblib.load('deployment/logistic_model.pkl')

# Sayfa başlığı
col1, col2, col3, col4= st.columns(4)

# İlk metrik
with col1:
    container1 = st.container(border=True)
    rmse = container1.metric(label="RMSE" , value="1.176", delta="-7.473")

# İkinci metrik
with col2:
    container2 = st.container(border=True)
    r2 = container2.metric(label="R2 Score" , value="0.983", delta="0.818")

with col3:
    container3 = st.container(border=True)
    r2 = container3.metric(label="Mean" , value="12.30", delta="-", delta_color="off")

with col4:
    container4 = st.container(border=True)
    r2 = container4.metric(label="STD" , value="9.44", delta="-", delta_color="off")


st.title('Delivery Time Prediction')

# Ana sayfada gösterilecek sekmeler
tab1, tab2, tab3 = st.tabs(["Product & Date", "Customer & Seller", "Cargo"])

with (tab1):
    payment_value = st.number_input("Total Price")
    quantity = st.number_input("Quantity", min_value=1 , max_value=5)
    time_box= datetime(2017, 1, 1)
    time = st.date_input("Date Purchased", time_box)
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
    special_days = st.selectbox("Special Days", special_day)
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
    c_state = ["CE" , "DF" , "ES" , "GO" , "MG" , "PE" , "PR" ,"RJ" , "RS" , "SC" , "SP", "BA", "Other"]
    customer_state = st.selectbox("Customer State" , c_state)
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
    cities_status = st.radio("Customer City Status", ["Big", "Major", "Medium", "Small"])
    if cities_status == "Big":
        cities_status = 0
    elif cities_status == "Major":
        cities_status = 1
    elif cities_status == "Medium":
        cities_status = 2
    else:
        cities_status = 3
    state = st.selectbox( "Seller State", ["PR", "RJ", "RS", "SC", "RP", "SP", "Other"])
    seller_state = ("Seller State", state)
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

with (tab3):
    prepare_time = st.number_input("Product Preparation Time" , step=1)
    product_weight_g = st.number_input("Product Weight", min_value=1)
    product_cm3 = st.number_input("Product cm3")
    cargo_score = st.selectbox("Shipping Company", ["A Class", "B Class", "C Class", "D Class", "E Class"])
    if cargo_score == "E Class":
        cargo_score = 25
    elif cargo_score == "D Class":
        cargo_score = 60
    elif cargo_score == "C Class":
        cargo_score = 100
    elif cargo_score == "B Class":
        cargo_score = 220
    else:
        cargo_score = 500

    distance_km = st.slider('Distance', min_value=10, max_value=8736,)
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

    if st.button('Predict'):
#        st.balloons()
        # Call the prediction function with input features
        predicted_score = predict_delivery_time(product_weight_g, payment_value, distance_km, cities_status,
                                               quantity, year, product_cm3, month, prepare_time, cargo_score, *season,
                                               *special_day, *weekday, *state, *c_state)

    try:
        if int(predicted_score) < 2:
            st.success("🚚 Your cargo will be delivered within 48 hours.")
        else:
            st.success(f'🚚 Estimated Delivery Time: {time + timedelta(int(predicted_score))}')
    except:
        st.write("")

html_sticky_footer = """
<style>
    body {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        margin: 0;
    }
    footer {
        padding: 20px;
        background-color: #f2f2f2);
        width: 100%;
        position: fixed;
        bottom: 0;
        left: 0; /* fix positioned left edge */
        box-sizing: border-box; /* fix padding issues */
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>

<footer>
    <div style="display: flex; align-items: center;">
        <div>
            <a href="https://kaggle.com" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-512.png" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        <a href="https://github.com/VeriBilimiProje" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        </div>
        <div style="margin-left: 10px;">
            <span style="font-size: 12px; color: #666;">Data Sapiens &copy;2024</span>
        </div>
    </div>
    <div>
        <a href="https://linktr.ee/mrakar" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://media.licdn.com/dms/image/D4D03AQGcvyGbq29esQ/profile-displayphoto-shrink_400_400/0/1713544700654?e=1719446400&v=beta&t=8rNFjSu46qxavynGcNQTUXZ4kDO7ewEf_TYxViYLi5s" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        <a href="https://github.com/umitdkara" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/154842224?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        <a href="https://github.com/ecan57" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/105751954?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a> 
        <a href="https://github.com/leylalptekin" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/48180024?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        <a href=" " style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://i.hizliresim.com/6uhz7is.png" style="width: 30px; height: 30px; border-radius: 50%;"></a>
    </div>
</footer>
"""

# Display the custom sticky footer
st.markdown(html_sticky_footer, unsafe_allow_html=True)

st.sidebar.image("deployment/assets/datasapienslogo.png")
