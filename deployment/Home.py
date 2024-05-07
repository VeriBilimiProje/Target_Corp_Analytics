import base64
import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# with open("deployment/style.css", "r", encoding="utf-8") as pred:
    # footer_html = f"""{pred.read()}"""
    # st.markdown(footer_html, unsafe_allow_html=True)

st.set_page_config(
    page_title="data sapiens",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'get help': 'https://www.extremelycoolapp.com/help',
        'report a bug': "https://www.extremelycoolapp.com/bug",
        'about': "# this is a header. this is an *extremely* cool app!"
    }
)

# st.image("deployment/assets/olist.png", width=200, use_column_width="never")

st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://i.hizliresim.com/c3v6sx3.png" alt="Olist Logo" width="200"/>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """

## Project Overview:
- The "Data Sapiens" dashboard showcases insights derived from a publicly available dataset of e-commerce orders from Olist, comprising 100,000 orders between 2016 and 2018.
- Leveraging machine learning models, namely Catboost and Logistic Regression, it provides valuable information about various aspects of the customer journey, including order status, checkout, and customer reviews.
- The dataset encompasses details about sellers listing products on Olist, along with customer behavior and demographic data.

---

### Models Utilized:
1. **Review Score Prediction:**
   - Model: Logistic Regression
   - Description: Predicts customer review scores based on various factors.
   
2. **Delivery Time Prediction:**
   - Model: Catboost
   - Description: Predicts delivery times for orders placed on Olist.

---

### Model Evaluation:
- **Review Score Prediction:**
  - Evaluation Metrics: Accuracy, ROC Auc , F1 Score
  - Performance: Achieved an accuracy of 0.881%, ROC Auc score of 0.794 and F1 score of 0.933%.
  
- **Delivery Time Prediction:**
  - Evaluation Metrics: RMSE, R2 Score
  - Performance: Achieved an RMSE of 1.176 and an R2 score of 0.983.

---
""", unsafe_allow_html=True)

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
        background-color: rgba(47, 69, 206, 0.3);
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
