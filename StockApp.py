import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from datetime import timedelta

def get_stock_data(stock_symbols, start_date, end_date):
    companies = [yf.Ticker(symbol) for symbol in stock_symbols]
    stock_data = [yf.download(symbol, start=start_date, end=end_date) for symbol in stock_symbols]
    summaries = [company.info['longBusinessSummary'] for company in companies]
    return companies, stock_data, summaries

def calculate_metrics(stock_data):
    returns = stock_data['Close'].pct_change()
    avg_return = returns.mean() * 252  # Assuming 252 trading days in a year
    risk = returns.std() * (252 ** 0.5)  # Annualized standard deviation
    sharpe_ratio = avg_return / risk
    return avg_return, risk, sharpe_ratio

# Streamlit sidebar widgets to allow user input
selected_stocks = st.sidebar.multiselect("Select Stock Symbols", ["AAPL", "MSFT", "TSLA", "GOOG"], default=["AAPL"])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2022-12-31"))

# Calculate the start date based on the end date
start_date = end_date - timedelta(days=35)  # Show last 35 days of data (1 month) for the plot
companies, stock_data, summaries = get_stock_data(selected_stocks, start_date, end_date)

# Displaying information, data info, and charts using Streamlit
metrics_data = []
portfolio_returns = []  # Initialize the portfolio_returns variable
portfolio_risks = []
for i, company in enumerate(companies):
    st.write(f"### {selected_stocks[i]}")
    st.markdown(f"**{company.info['longName']}**")
    st.write(summaries[i][:200] + "...")
    st.write("[Read more](#)")  # Placeholder link to be replaced later with a real link

    # Displaying the stock data within the chosen dates
    st.write(stock_data[i].tail())

    # Calculate metrics for the stock
    avg_return, risk, sharpe_ratio = calculate_metrics(stock_data[i])
    metrics_data.append([selected_stocks[i], avg_return, risk, sharpe_ratio])

    # Calculate portfolio returns and risks
    portfolio_returns.append(avg_return)
    portfolio_risks.append(risk)

    # Plotting the line chart for the last 1 month
    last_1_month_data = stock_data[i].iloc[-35:]
    st.line_chart(last_1_month_data["Close"])

# Creating a DataFrame to display the calculated metrics
metrics_df = pd.DataFrame(metrics_data, columns=['Stock Symbol', 'Avg. Annual Return', 'Risk', 'Sharpe Ratio'])
st.write("## Individual Stock Metrics")
st.write(metrics_df)

# Calculate the combined metrics for the portfolio
portfolio_avg_return = sum(portfolio_returns) / len(portfolio_returns)
portfolio_risk = (sum([r**2 for r in portfolio_risks]) / len(portfolio_risks))**0.5
portfolio_sharpe_ratio = portfolio_avg_return / portfolio_risk

# Displaying the combined metrics for the portfolio
st.write("## Portfolio Metrics")
st.write("Average Annual Return:", portfolio_avg_return)
st.write("Risk:", portfolio_risk)
st.write("Sharpe Ratio:", portfolio_sharpe_ratio)
