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
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
import datetime as dt
from statsmodels.tsa.stattools import adfuller

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

symbol='BTC-USD'
st.header('Bitcoin')
st.image('bitcoin.jpg')
st.write('Bitcoin uses peer-to-peer technology to operate with no central authority or banks; managing transactions and the issuing of bitcoins is carried out collectively by the network.')
start = st.date_input('Start',dt.date(2021,8, 12))
end=st.date_input('End',value=pd.to_datetime('today'),key=1)
df = yf.download(symbol,start,end)
st.subheader('Dates of BTC-USD stock')
df=df.tail(15)
st.table(df)

st.subheader('Calculating and describing mean,std,count')
df1=df.describe()
st.table(df1)

new_df = pd.DataFrame({symbol : df['Open'], symbol : df['Close']})
st.subheader('Setting date as index')
new_df1 = pd.DataFrame({symbol : df['Open'], symbol : df['Close']})
df_ts1=new_df1.tail(3)
st.table(df_ts1)

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure(figsize=(13, 5))
plt.plot(pd.DataFrame({symbol : df['Close']}))
st.subheader('Plotting the Close data')
st.pyplot(plt)
plt.close()

def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        st.write('Critical values for 1%,5%,10%:-',dfoutput)
        
        
ts = pd.DataFrame({symbol : df['Close']})
test_stationarity(ts)
     
rolmean = ts.rolling(window=12).mean()
rolvar = ts.rolling(window=12).std()
plt.figure(figsize=(13, 5))
plt.plot(ts, label='Original')
plt.plot(rolmean, label='Rolling Mean')
plt.plot(rolvar, label='Rolling Standard Variance')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)
st.subheader('Plotting the Rolling Mean & Variance and find Insights')
st.pyplot(plt)
plt.close()

decomposition = sm.tsa.seasonal_decompose(ts, model='multiplicative')
fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
st.subheader('Vanila decomposition of multiplicative time series')
plt.show()
st.pyplot(plt)
plt.close()

st.subheader('After resampling')
df_ts_1= df_ts.resample('M').mean()
df_ts_1 = pd.DataFrame({symbol : df['Close']})
test_stationarity(df_ts_1)

st.subheader('Log values')
tsmlog = np.log10(df_ts_1)
tsmlog.dropna(inplace=True)

tsmlogdiff = tsmlog.diff(periods=1)
tsmlogdiff.dropna(inplace=True)
test_stationarity(tsmlogdiff)

crypto_data = {}
crypto_data['bitcoin'] = pd.read_csv('bitcoin_price.csv', parse_dates=['Date'])

df_bitcoin = pd.DataFrame(crypto_data['bitcoin'])
df_bitcoin = df_bitcoin[['Date','Close']]
df_bitcoin.set_index('Date', inplace = True)
st.subheader('Took a csv file of bitcoin prices in 2018')

# fit model
model = ARIMA(df_bitcoin, order=(5,1,0))
model_fit = model.fit()
summary=model_fit.summary()
st.write(summary)
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
st.subheader('Fitting the model using ARIMA(statsmodel)')
st.pyplot(plt)
plt.close()
residuals.plot(kind='kde')
plt.show()
st.subheader('Residuals while plotting')
st.pyplot(plt)
plt.close()

X = df_bitcoin.values
size = int(len(X) * 0.80)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
st.subheader('Predicting future values by graph of BTC-USD')
st.pyplot(plt)
plt.close()
