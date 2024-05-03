import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import seaborn as sns
import numpy as np

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

st.sidebar.image("deployment/assets/datasapienslogo.png", use_column_width=True)

# st.title('DATA SAPIENS')
page_bg = '''
<style>
body {
background-image: url("https://example.com/background.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg, unsafe_allow_html=True)

# Orta noktaya esneklik ekleyerek kutuyu düzenle
st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
        <div style="background-color: rgba(255, 255, 255, 0.5); padding: 20px; border-radius: 10px; max-width: 50%;">
            <h1 style="color: #000000; text-align: center;">E-Commerce Prediction</h1>
            <p style="color: #000000; text-align: center;"> Bu proje, Olist Store'da yapılan siparişlerin Brezilya e-ticaret genel veri kümesidir. Veri kümesi, 2016'dan 2018'e kadar Brezilya'daki birden fazla pazar yerinde yapılan 100 bin siparişin bilgisine sahiptir.
## 🎢Müşteri Memnuniyeti Modelleme:  

Projemizde müşteri memnuniyetine etki eden faktörleri belirleyip memnuniyet tahmini modelini geliştirdik.
## 🎢Modelleme Yöntemleri:  

Regresyon analizi: Regresyon analizi, bağımlı bir değişken (örneğin, müşteri memnuniyeti puanı) ile bir veya birden fazla bağımsız değişken (örneğin, ürün fiyatı, teslimat süresi) arasındaki ilişkiyi modellemek için kullanılır.  
## 📑Modelleme Değerlendirme:  

Ortalama mutlak hata (MAE): MAE, tahmin edilen değerler ile gerçek değerler arasındaki ortalama mutlak farkı ölçer.  
Ortalama karesel hata (MSE): MSE, tahmin edilen değerler ile gerçek değerler arasındaki ortalama karesel farkı ölçer.  
# 🔍Sonuç:  

Olist veri seti, müşteri memnuniyeti modelleme için değerli bir kaynaktır. Bu veri seti, siparişler, ürünler, müşteriler ve değerlendirmeler gibi çeşitli bilgileri içerir. Bu bilgiler, müşteri memnuniyetini etkileyen faktörleri belirlemek ve tahmin modelleri oluşturmak için kullanılabilir.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
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
        background-color: #f2f2f2;
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
            <a href="https://linktr.ee/mrakar" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://media.licdn.com/dms/image/D4D03AQGcvyGbq29esQ/profile-displayphoto-shrink_400_400/0/1713544700654?e=1719446400&v=beta&t=8rNFjSu46qxavynGcNQTUXZ4kDO7ewEf_TYxViYLi5s" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        <a href="https://github.com/umitdkara" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/154842224?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        </div>
        <div style="margin-left: 10px;">
            <span style="font-size: 12px; color: #666;">data sapiens &copy;2024</span>
        </div>
    </div>
    <div>
        <a href="https://linktr.ee/mrakar" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://media.licdn.com/dms/image/D4D03AQGcvyGbq29esQ/profile-displayphoto-shrink_400_400/0/1713544700654?e=1719446400&v=beta&t=8rNFjSu46qxavynGcNQTUXZ4kDO7ewEf_TYxViYLi5s" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        <a href="https://github.com/umitdkara" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/154842224?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        <a href="https://github.com/ecan57" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/105751954?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a> 
        <a href="https://github.com/leylalptekin" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/48180024?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a>
    </div>
</footer>
"""

# Display the custom sticky footer
st.markdown(html_sticky_footer, unsafe_allow_html=True)

html("""
<style>
    @import url('./style.css'); 
</style>
""")