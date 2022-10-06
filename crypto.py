import streamlit as st
import yfinance as yf
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import math
import plotly.graph_objs as go
import statistics as sp
import scipy.stats as stats
from scipy.stats.mstats import gmean
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import expon
import seaborn as sns


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