import streamlit as st
from streamlit import components
import pandas as pd
import numpy as np
import plotly.express as px
import base64

df = pd.read_csv('datasets/final_dataset.csv')

st.set_page_config(
    page_title="Data Sapiens",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.sidebar.image("deployment/assets/logo.png", use_column_width=True)

html_code = """
<div style="display: flex;">
    <a href="https://linktr.ee/mrakar" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 100px; height: 100px; overflow: hidden; border-radius: 50%; border: 2px solid black;">
            <img src="https://media.licdn.com/dms/image/D4D03AQGcvyGbq29esQ/profile-displayphoto-shrink_400_400/0/1713544700654?e=1719446400&v=beta&t=8rNFjSu46qxavynGcNQTUXZ4kDO7ewEf_TYxViYLi5s" style="width: 100%; height: auto;">
        </div>
    <div style="text-align: center;">Muhammed AKAR</div>
    </a>
    <a href="https://github.com/umitdkara" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 100px; height: 100px; overflow: hidden; border-radius: 50%; border: 2px solid black;">
            <img src="https://avatars.githubusercontent.com/u/154842224?v=4" style="width: 100%; height: auto;">
        </div>
    <div style="text-align: center;">Ümit KARA</div>
    </a>
    <a href="https://github.com/ecan57" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 100px; height: 100px; overflow: hidden; border-radius: 50%; border: 2px solid black;">
            <img src="https://avatars.githubusercontent.com/u/105751954?v=4" style="width: 100%; height: auto;">
        </div>
    <div style="text-align: center;">Emine CAN</div>
    </a>
    <a href="https://github.com/leylalptekin" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 100px; height: 100px; overflow: hidden; border-radius: 50%; border: 2px solid black;">
            <img src="https://avatars.githubusercontent.com/u/48180024?v=4" style="width: 100%; height: auto;">
        </div>
    <div style="text-align: center;">Leyla ALPTEKİN</div>
    </a>
</div>

"""

st.title('DATA SAPIENS')
st.markdown(html_code, unsafe_allow_html=True)
'---'
st.subheader('Grup Marşımız')
st.audio("deployment/assets/datasapiens.mp3", format="audio/mpeg")
'---'

with open("deployment/assets/main.md", "r", encoding="utf-8") as file:
    markdown_text = file.read()

st.markdown(markdown_text)

st.subheader('Proje Dosyaları')

st.markdown('''
<div style="display: flex;">
    <a href="https://linktr.ee/mrakar" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 75px; height: 75px; overflow: hidden; border-radius: 50%; padding:10px;order: 2px solid black;">
            <img src="https://cdn.iconscout.com/icon/free/png-512/free-kaggle-3521526-2945029.png?f=webp&w=512" style="width: 100%; height: auto;">
        </div>
        <div style="text-align: center;">Kaggle</div>
    </a>
       <a href="https://linktr.ee/mrakar" style="margin-right: 10px;text-decoration: none; color: black;">
        <div style="width: 75px; height: 75px; overflow: hidden; border-radius: 50%;padding:10px;order: 2px solid black;">
            <img src="https://cdn.iconscout.com/icon/free/png-512/free-github-159-721954.png?f=webp&w=512" style="width: 100%; height: auto;">
        </div>
        <div style="text-align: center;">GitHub</div>
    </a>
''', unsafe_allow_html=True)