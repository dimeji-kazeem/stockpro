import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import streamlit as st

def get_stock_data(stock_symbol):
    company = yf.Ticker(stock_symbol)
    return company

apl = get_stock_data("AAPL")
mst = get_stock_data("MSFT")
tsl = get_stock_data("TSLA")
gog = get_stock_data("GOGL")

apple = yf.download("AAPL", start="2015-01-01", end="2022-12-31")
microsoft = yf.download("MSFT", start="2015-01-01", end="2022-12-31")
tesla = yf.download("TSLA", start="2015-01-01", end="2022-12-31")
google = yf.download("GOGL", start="2015-01-01", end="2022-12-31")

#Fetching the historical data by valid periods
dt1 = apl.history(period="5y")
dt2 = mst.history(period="5y")
dt3 = tsl.history(period="5y")
dt4 = gog.history(period="5y")

#Markdown
st.write(""" ### Apple""")
#Detailed summary about apple
st.write(apl.info['longBusinessSummary'])
#DataFrame
st.write(apple)
#Charting
st.line_chart(dt1.values)

