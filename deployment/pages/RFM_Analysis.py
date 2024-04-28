import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)
def eda():
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("==============Currenly run EDA option!==============\n")

    st.title("Olist E-Commerce Dataset EDA :sparkles:")
    st.header('Brazilian E-Commerce EDA', divider='rainbow')
    st.subheader('Daily Orders')
