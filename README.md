## 💆‍♀️About E-Commerce Customer Satisfaction and Influencing Factor Estimation Project💆‍♂️

<a target="_blank" href="https://datasapiens.streamlit.app/"><img src="https://i.pinimg.com/originals/16/f2/73/16f27340e4def8cf891d80e0645b9e4c.png"></img></a>

Overview

This project aims to predict customer satisfaction and identify the factors that influence it using an e-commerce dataset. The dataset includes various features such as customer demographics, order details, payment information, and product characteristics. By analyzing this data, we aim to build machine learning models to predict customer satisfaction and provide insights into the key drivers of customer satisfaction.

👉Table of Contents

1-Overview

2-Dataset

3-Feature Engineering

4-Modeling

5-Results

6-Conclusion

7-How to Use

8-License

2-DATASET 📝

Our data set consists of 36 variables and over 100 thousand observations. olist e-commerce websites in Brazil cover the years 2016-2018.
Some of these variables:

-customer_city: City of the customer                                         

-order_status: Status of the order.

-order_purchase_timestamp: Date and time of the order purchase.   

-order_approved_at: Date and time when the order was approved.

-order_delivered_carrier_date: Date and time when the order was handed to the carrier.

-order_delivered_customer_date: Date and time when the order was delivered to the customer.

-order_estimated_delivery_date: Estimated delivery date of the order.

-review_score: Review score given by the customer.

3-Feature Engineering📑
Date Features
Extracted year, month, day, hour, and minute from date columns.
Calculated time differences between various order stages (e.g., approval time, delivery time).
Categorical Features
Applied one-hot encoding to categorical features such as payment type and order status.
Mapped product categories to higher-level categories.
Numerical Features
Scaled numerical features using RobustScaler.

1. **Date Features:**
   - Extracted year, month, day, hour, and minute from date columns.
   - Calculated time differences between various order stages (e.g., approval time, delivery time).
2. **Categorical Features:**
   - Applied one-hot encoding to categorical features such as payment type and order status.
   - Mapped product categories to higher-level categories.
3. **Numerical Features:**
   - Scaled numerical features using RobustScaler.

