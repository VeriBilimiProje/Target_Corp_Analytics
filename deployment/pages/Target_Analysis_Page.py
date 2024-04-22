import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)

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

st.title("Data Analysis by Customer Satisfaction")
'---'

df = pd.read_csv('datasets/final_dataset.csv')

################################################################
# Urunun fiyat/kargo fiyati oraninin ortalama review_score
################################################################


df_glad_by_year = df.groupby('month_year')['review_score'].mean().reset_index()

fig = px.line(df_glad_by_year, x='month_year', y='review_score', title='Tarih Değişkenine Göre Line Chart',)
fig.update_layout(
        xaxis_title='Tarih',
        yaxis_title='Memnuniyet Ortalaması',

    )

st.plotly_chart(fig)

st.markdown('''
    What is Lorem Ipsum?
    Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

    Why do we use it?
    It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).


    ''')

################################################################
# Urunun fiyat/kargo fiyati oraninin ortalama review_score
################################################################


df_satis = pd.DataFrame(df.groupby(['review_score'])['price_freight_ratio'].mean().reset_index())

fig_satis = px.bar(df_satis, x='review_score', y='price_freight_ratio')

fig_satis.update_layout(
        xaxis_title='Review Score',
        yaxis_title='Freight Value/Price',
        yaxis_range=[0.3, df_satis['price_freight_ratio'].max()]
    )
st.plotly_chart(fig_satis)
st.markdown('''
   What is Lorem Ipsum?
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

Why do we use it?
It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).


    ''')
################################################################
# Musterinin bekleme suresine gore ortalama review_score
################################################################

df_wait = pd.DataFrame(df.groupby(['review_score'])['customer_wait_time(day)'].mean().reset_index())

fig_wait = px.bar(df_wait, x='review_score', y='customer_wait_time(day)')

fig_wait.update_layout(
    xaxis_title='Review Score',
    yaxis_title='Customer Wait Days',
    yaxis_range=[5, df_wait['customer_wait_time(day)'].max()]

)
st.plotly_chart(fig_wait)

################################################################
# Odeme yontemlerine gore ortalama review_score
################################################################

df_pay = pd.DataFrame(df.groupby(['payment_type'])['review_score'].mean().reset_index())

fig_pay = px.bar(df_pay, x='payment_type', y='review_score')

fig_pay.update_layout(
    xaxis_title='Payment Type',
    yaxis_title='Review Score',
    yaxis_range=[3.75, df_pay['review_score'].max()]

)
st.plotly_chart(fig_pay)

################################################################
# Kategorilere gore ortalama review_score
################################################################

df_cat = pd.DataFrame(df.groupby(['category'])['review_score'].mean().reset_index())

fig_cat = px.bar(df_cat, x='category', y='review_score')

fig_cat.update_layout(
    xaxis_title='Categories',
    yaxis_title='Review Score'
)
st.plotly_chart(fig_cat)

################################################################
# Kargo Fiyatlari Histogram
################################################################

fig_hist = px.histogram(df, x='freight_value', title='Freight Value Histogram')

fig_hist.update_layout(
    xaxis_title='Freight Value',
    yaxis_title='Count'
)

st.plotly_chart(fig_hist)

################################################################
