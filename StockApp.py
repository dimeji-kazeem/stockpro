import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

def get_stock_data(stock_symbols, start_date, end_date):
    companies = [yf.Ticker(symbol) for symbol in stock_symbols]
    stock_data = [yf.download(symbol, start=start_date, end=end_date) for symbol in stock_symbols]
    summaries = [company.info['longBusinessSummary'] for company in companies]
    return companies, stock_data, summaries

# Streamlit sidebar widgets to allow user input
selected_stocks = st.sidebar.multiselect("Select Stock Symbols", ["AAPL", "MSFT", "TSLA", "GOOG"], default=["AAPL"])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2022-12-31"))

companies, stock_data, summaries = get_stock_data(selected_stocks, start_date, end_date)

# Displaying information and charts using Streamlit
for i, company in enumerate(companies):
    st.write(f"### {selected_stocks[i]}")
    st.markdown(f"**{company.info['longName']}**")
    st.write(summaries[i][:200] + "...")
    st.write("[Read more](#)")  # Placeholder link to be replaced later with a real link
    st.write(stock_data[i])

    # Plotting the line chart
    st.line_chart(stock_data[i]["Close"])
