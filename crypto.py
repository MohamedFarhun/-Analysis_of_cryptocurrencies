import streamlit as st
import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
from pandas import DataFrame
import matplotlib.pyplot as plt
from pandas import datetime
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

st.title('Analysis of cryptocurrencies')

def cryptocurrency():
    """Analysis_of_cryptocurrencies.cryptocurrency().value"""
    global cryptocurrency
    if cryptocurrency: return cryptocurrency
    ...

with st.sidebar:
    st.header("Analysis of cryptocurrencies-Round 2")
    st.image("cryptocurrency.jpg")
    st.header("Team name : TEKKYZZ")
    st.write("Leader     : MOHAMED FARHUN M")
    st.write("Member 1   : NANDHAKUMAR S")
    st.write("Member 2   : DHIVAKAR S")
    st.subheader("How do cryptocurrencies work?")
    video_file = open('Crypto.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

symbol = 'BTC-USD'
start=st.date_input('Start',value =pd.to_datetime('2022-10-01'))
end=st.date_input('End',value=pd.to_datetime('today'))
df = yf.download(symbol,start,end)
df=df.head(5)
st.table(df)

df1=df.describe()
st.table(df1)

df2 = pd.DataFrame({symbol : df['Date'], symbol : df['Close']}
analysis=df2.head(10)
st.table(analysis)
