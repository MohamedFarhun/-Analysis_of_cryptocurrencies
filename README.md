# ARIMA statsmodel-Cryptocurrency Analysis
## This app gives us the analysis of various cryptocurrencies.
We have developed an app which predicts the future cryptocurrency values of btc-usd,eth-usd.Took the csv file of bitcoin and found the rolling mean,variance.Plotted the vanila decomposition graph for cryptos.
### Team Details:
#### Team Name: TEKKYZ
#### Team Leader: MOHAMED FARHUN M 
#### Members: NANDHAKUMAR S,DHIVAKAR S
#### Designation: Student at BANNARI AMMAN INSTITUTE OF TECHNOLOGY, SATHYAMANGALAM.
#### Contact: +919360593132
#### Mail: mohamedfarhun.it20@bitsathy.ac.in

## What we used......
• Daisi


• Streamlit UI


• pandas


• sklearn


• yfinance


• Statsmodel


#### We have done our project in the daisi platform (https://app.daisi.io/daisies/farhun/Analysis%20of%20cryptocurrencies/app) using Streamlit UI. We used yfinance-Yahoo finance(fetches live dataset of cryptos) from which we can have a statistical analysis of cryptos(graphically).
## Features
•Predicting Cryptocurrency graph variations from the start and end date


•predicts the future of cryptocurrency graph.


## Our daisi app

It is recommended to use this application on the daisi platform itself using the link https://app.daisi.io/daisies/farhun/Analysis%20of%20cryptocurrencies/app
However, you can still use your own editor using the below method:

### Python
See the docs for pyDaisi installation and authentication.

### Calling our app
import pydaisi as pyd
analysis_of_cryptocurrencies = pyd.Daisi("farhun/Analysis of cryptocurrencies")

### Documented endpoints
##### Cryptocurrency


analysis_of_cryptocurrencies.cryptocurrency().value

##### Analysis_of_cryptocurrency

analysis_of_cryptocurrencies.analysis_of_cryptocurrency().value

### R
library(rdaisi)

configure_daisi(python_path="/usr/local/bin/python3")

analysis_of_cryptocurrencies <- Daisi("farhun/Analysis of cryptocurrencies")

### Endpoints
##### Cryptocurrency

analysis_of_cryptocurrencies$cryptocurrency()$value()

##### App
analysis_of_cryptocurrencies$app()$value()

#### st_ui
analysis_of_cryptocurrencies$st_ui()$value()


### Screenshot of our app
![image](https://user-images.githubusercontent.com/86124759/195563159-b7da0b33-90cf-4f66-b67b-265ff723e062.png)
![image](https://user-images.githubusercontent.com/86124759/195563332-db10fea3-edbd-4d4a-b57d-fa7e08495ef5.png)

