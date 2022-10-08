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
import datetime as dt

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
start = st.date_input('Start',dt.date(2021,8, 12))
end=st.date_input('End',value=pd.to_datetime('today'),key=1)
df = yf.download(symbol,start,end)
df=df.head(5)
st.table(df)

symbol = 'BTC-USD'
start = st.date_input('Start',dt.date(2021,8, 13))
end=st.date_input('End',value=pd.to_datetime('today'),key=2)
df = yf.download(symbol,start,end)
dff=df.tail(5)
st.table(dff)

df1=df.describe()
st.table(df1)

new_df = pd.DataFrame({symbol : df['Open'], symbol : df['Close']})
df_ts=new_df.head(3)
st.table(df_ts)
new_df1 = pd.DataFrame({symbol : df['Open'], symbol : df['Close']})
df_ts1=new_df1.tail(3)
st.table(df_ts1)

st.set_option('deprecation.showPyplotGlobalUse', False)
df_ts=plt.plot()
st.pyplot(df_ts)
