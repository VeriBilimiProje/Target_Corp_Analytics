import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Add joblib library
from datetime import datetime, timedelta

# Load the pickled model
# model = joblib.load('deployment/log_reg.pkl')

# Sayfa başlığı
st.title('Delivery Time Prediction')