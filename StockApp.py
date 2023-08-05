import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from datetime import timedelta
import numpy as np

def get_stock_data(stock_symbols, start_date, end_date):
    companies = [yf.Ticker(symbol) for symbol in stock_symbols]
    stock_data = [yf.download(symbol, start=start_date, end=end_date) for symbol in stock_symbols]
    adj_close_values = [data['Adj Close'] for data in stock_data]
    summaries = [company.info['longBusinessSummary'] for company in companies]

    return companies, stock_data, adj_close_values, summaries

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

# Fetch stock data and summaries
companies, stock_data, adj_close_values, summaries = get_stock_data(selected_stocks, start_date, end_date)

tab1, tab2, tab3, tab4 = st.tabs([":moneybag: About", ":chart: Component Stocks", ":bar_chart: Portfolio Metrics", ":chart_with_upwards_trend: Perfomance Prediction"])

# Tab 1: Last five records and line chart over the last one month to the end date
with tab1:
    for i, company in enumerate(companies):
        st.write(f"### {selected_stocks[i]}")
        st.markdown(f"**{company.info['longName']}**")
        st.write(summaries[i][:200] + "...")
        st.write("[Read more](#)")  # Placeholder link to be replaced later with a real link

        # Displaying the stock data within the chosen dates
        st.write(stock_data[i].tail())

        # Plotting the line chart for the last 1 month
        last_1_month_data = stock_data[i].iloc[-35:]
        st.line_chart(last_1_month_data["Adj Close"])

# Tab 2: Stock Metrics
with tab2:
    # Create a DataFrame to display stock metrics
    metrics_data = []
    for i, company in enumerate(companies):
        stock_symbol = selected_stocks[i]
        adj_close = adj_close_values[i].values

        # Calculate metrics for the stock
        returns = round(stock_data[i]['Adj Close'].pct_change(), 3)
        avg_return, risk, sharpe_ratio = calculate_metrics(stock_data[i])

        # Extract financial metrics from company.info dictionary
        market_cap = company.info.get('marketCap', None)
        eps = company.info.get('trailingEps', None)
        pe_ratio = company.info.get('trailingPE', None)
        dividend_yield = company.info.get('trailingAnnualDividendYield', None)
        dividend_payout_ratio = company.info.get('payoutRatio', None)
        roe = company.info.get('returnOnEquity', None)
        beta = company.info.get('beta', None)
        volatility = round(returns.std(), 3)  # Volatility as daily standard deviation of returns
        debt_to_equity = company.info.get('debtToEquity', None)
        cagr = round(((adj_close[-1] / adj_close[0]) ** (252 / len(adj_close))) - 1, 3)

        # Revenue Growth and Profit Margin data might not be available in 'company.info'
        revenue_growth = None
        profit_margin = None
        if 'financialData' in company.info:
            financial_data = company.info['financialData']
            revenue_growth = round(financial_data.get('revenueGrowth', None), 3)
            profit_margin = round(financial_data.get('profitMargins', None), 3)

        metrics_data.append([stock_symbol, round(adj_close[-1], 3), avg_return, risk, market_cap, eps, pe_ratio, dividend_yield,
                             dividend_payout_ratio, roe, beta, volatility, debt_to_equity, cagr, revenue_growth, profit_margin])

    stock_metrics_df = pd.DataFrame(metrics_data, columns=['Stock Symbol', 'Adj. Close', 'Avg. Annual Return', 'Risk', 'Market Cap',
                                                           'EPS', 'P/E Ratio', 'Dividend Yield', 'Dividend Payout Ratio',
                                                           'Return on Equity', 'Beta', 'Volatility', 'D/E Ratio', 'CAGR',
                                                           'Revenue Growth', 'Profit Margin'])
    st.write(stock_metrics_df.T)
    
    # Line chart for all selected stocks - Last 1 Month
    st.write("## Line Chart - Last 1 Month")
    fig_month = go.Figure()
    for i, symbol in enumerate(selected_stocks):
        stock_data_plot = stock_data[i].loc[start_date:end_date]
        fig_month.add_trace(go.Scatter(x=stock_data_plot.index, y=stock_data_plot['Adj Close'], name=symbol))

    fig_month.update_layout(title_text='Stock Prices - Last 1 Month',
                            xaxis_title='Date',
                            yaxis_title='Adj Close Price',
                            xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_month)

    # Line chart for all selected stocks - Last 5 Years
    st.write("## Line Chart - Last 5 Years")
    fig_5_years = go.Figure()
    for i, symbol in enumerate(selected_stocks):
        stock_data_plot = stock_data[i].loc[end_date - pd.DateOffset(years=5):end_date]
        fig_5_years.add_trace(go.Scatter(x=stock_data_plot.index, y=stock_data_plot['Adj Close'], name=symbol))

    fig_5_years.update_layout(title_text='Stock Prices - Last 5 Years',
                              xaxis_title='Date',
                              yaxis_title='Adj Close Price',
                              xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_5_years)
    

def calculate_portfolio_metrics(stock_data, portfolio_weights):
    portfolio_returns = []
    for i, data in enumerate(stock_data):
        returns = data['Close'].pct_change().dropna()
        portfolio_returns.append(returns)

    # Calculate portfolio returns
    portfolio_return = sum([w * ret for w, ret in zip(portfolio_weights, portfolio_returns)])

    # Calculate ROI
    initial_investment = 100000  # Initial investment amount
    final_portfolio_value = initial_investment * (1 + portfolio_return[-1])
    roi = (final_portfolio_value - initial_investment) / initial_investment

    # Calculate CAGR
    cagr = (final_portfolio_value / initial_investment) ** (1 / len(portfolio_return)) - 1

    # Calculate standard deviation
    portfolio_std_dev = sum([w * ret.std() for w, ret in zip(portfolio_weights, portfolio_returns)])

    # Calculate sharpe ratio
    risk_free_rate = 0.02  # Assumed annual risk-free rate
    sharpe_ratio = (cagr - risk_free_rate) / portfolio_std_dev

    # Calculate beta
    benchmark_data = yf.download("^GSPC", start=start_date, end=end_date)['Close'].pct_change().dropna()
    covariance_matrix = pd.concat([portfolio_returns, benchmark_data], axis=1).cov()
    beta = covariance_matrix.iloc[0, 1] / benchmark_data.var()

    # Assuming the risk-free rate is the same as the benchmark
    treynor_ratio = (cagr - risk_free_rate) / beta

    # Calculate sortino ratio
    negative_returns = [ret[ret < 0] for ret in portfolio_returns]
    downside_std_dev = sum([w * neg_ret.std() for w, neg_ret in zip(portfolio_weights, negative_returns)])
    sortino_ratio = (cagr - risk_free_rate) / downside_std_dev

    # Calculate information ratio
    benchmark_returns = benchmark_data.values
    tracking_error = (portfolio_return - benchmark_returns).std()
    information_ratio = (cagr - risk_free_rate) / tracking_error

    # Calculate maximum drawdown
    rolling_max = (portfolio_return + 1).cummax()
    drawdown = (portfolio_return + 1) / rolling_max - 1
    max_drawdown = drawdown.min()

    return roi, cagr, portfolio_std_dev, sharpe_ratio, beta, treynor_ratio, sortino_ratio, information_ratio, max_drawdown

# Tab 3: Portfolio Summary
with tab3:
    # Calculate the combined metrics for the portfolio
    portfolio_avg_return = round(stock_metrics_df['Avg. Annual Return'].mean(), 3)
    portfolio_risk = round(stock_metrics_df['Risk'].mean(), 3)

    

    # Displaying the combined metrics for the portfolio
    st.write("## Portfolio Metrics")
    st.write("Average Annual Return:", portfolio_avg_return)
    st.write("Risk:", portfolio_risk)

    # DataFrame for portfolio metrics at enddate
    portfolio_metrics_enddate = stock_metrics_df.iloc[-1:][['Adj. Close', 'Avg. Annual Return', 'Risk', 'Market Cap', 'EPS',
                                                            'P/E Ratio', 'Dividend Yield', 'Dividend Payout Ratio',
                                                            'Return on Equity', 'Beta', 'Volatility', 'D/E Ratio', 'CAGR',
                                                            'Revenue Growth', 'Profit Margin']].reset_index(drop=True)

    #st.write("## Portfolio Metrics at End Date")
    #st.table(portfolio_metrics_enddate)
    
def calculate_portfolio_metrics(stock_data, portfolio_weights):
    portfolio_returns = []
    for i, data in enumerate(stock_data):
        returns = data['Close'].pct_change().dropna()
        portfolio_returns.append(returns)

    # Calculate portfolio returns
    portfolio_return = sum([w * ret for w, ret in zip(portfolio_weights, portfolio_returns)])

    # Calculate ROI
    initial_investment = 100000  # Initial investment amount
    final_portfolio_value = initial_investment * (1 + portfolio_return[-1])
    roi = (final_portfolio_value - initial_investment) / initial_investment

    # Calculate CAGR
    cagr = (final_portfolio_value / initial_investment) ** (1 / len(portfolio_return)) - 1

    # Calculate standard deviation
    portfolio_std_dev = sum([w * ret.std() for w, ret in zip(portfolio_weights, portfolio_returns)])

    # Calculate sharpe ratio
    risk_free_rate = 0.02  # Assumed annual risk-free rate
    sharpe_ratio = (cagr - risk_free_rate) / portfolio_std_dev

    # Calculate beta
    benchmark_data = yf.download("^GSPC", start=start_date, end=end_date)['Close'].pct_change().dropna()
    covariance_matrix = pd.concat([pd.DataFrame(portfolio_return), benchmark_data], axis=1).cov()
    beta = covariance_matrix.iloc[0, 1] / benchmark_data.var()

    # Assuming the risk-free rate is the same as the benchmark
    treynor_ratio = (cagr - risk_free_rate) / beta

    # Calculate sortino ratio
    negative_returns = [ret[ret < 0] for ret in portfolio_returns]
    downside_std_dev = sum([w * neg_ret.std() for w, neg_ret in zip(portfolio_weights, negative_returns)])
    sortino_ratio = (cagr - risk_free_rate) / downside_std_dev

    # Calculate information ratio
    benchmark_returns = benchmark_data.values
    tracking_error = (portfolio_return - benchmark_returns).std()
    information_ratio = (cagr - risk_free_rate) / tracking_error

    # Calculate maximum drawdown
    rolling_max = (portfolio_return + 1).cummax()
    drawdown = (portfolio_return + 1) / rolling_max - 1
    max_drawdown = drawdown.min()

    return roi, cagr, portfolio_std_dev, sharpe_ratio, beta, treynor_ratio, sortino_ratio, information_ratio, max_drawdown
    
    
# Tab 3: Portfolio Summary
with tab3:
    # Calculate portfolio metrics
    portfolio_weights = [1 / len(companies)] * len(companies)
    roi, cagr, portfolio_std_dev, sharpe_ratio, beta, treynor_ratio, sortino_ratio, information_ratio, max_drawdown = calculate_portfolio_metrics(stock_data, portfolio_weights)

    # Display additional portfolio metrics
    portfolio_metrics_enddate.loc[0, 'ROI'] = round(roi, 3)
    portfolio_metrics_enddate.loc[0, 'CAGR'] = round(cagr, 3)
    portfolio_metrics_enddate.loc[0, 'Standard Deviation'] = round(portfolio_std_dev, 3)
    portfolio_metrics_enddate.loc[0, 'Sharpe Ratio'] = round(sharpe_ratio, 3)
    portfolio_metrics_enddate.loc[0, 'Beta'] = round(beta, 3)
    portfolio_metrics_enddate.loc[0, 'Treynor Ratio'] = round(treynor_ratio, 3)
    portfolio_metrics_enddate.loc[0, 'Sortino Ratio'] = round(sortino_ratio, 3)
    portfolio_metrics_enddate.loc[0, 'Information Ratio'] = round(information_ratio, 3)
    portfolio_metrics_enddate.loc[0, 'Max Drawdown'] = round(max_drawdown, 3)

    st.write("## Portfolio Metrics at End Date")
    st.table(portfolio_metrics_enddate)
    
    # Line chart for portfolio monthly returns - Last 1 Year
    st.write("## Line Chart - Portfolio Monthly Returns - Last 1 Year")
    portfolio_returns = pd.concat([pd.Series(stock_data[i]['Close'].pct_change().dropna(), name=selected_stocks[i]) for i in range(len(companies))], axis=1)
    portfolio_returns['Portfolio'] = portfolio_returns.mean(axis=1)
    portfolio_returns_monthly = portfolio_returns.resample('M').agg(lambda x: (x + 1).prod() - 1)

    # Convert the dates to abbreviations (JAN, FEB, MAR, etc.)
    months_abbr = [dt.strftime('%b') for dt in portfolio_returns_monthly.index]

    fig_portfolio_returns = go.Figure()
    fig_portfolio_returns.add_trace(go.Scatter(x=months_abbr, y=portfolio_returns_monthly['Portfolio'], name="Portfolio Monthly Returns", mode='markers+lines'))

    # Calculate stock contributions to the portfolio
    total_contribution = sum([w * r for w, r in zip(portfolio_weights, portfolio_returns.mean())])
    stock_contributions_percent = [(w * r) / total_contribution * 100 for w, r in zip(portfolio_weights, portfolio_returns.mean())]
    stock_contributions = pd.DataFrame({"Stock Symbol": selected_stocks, "Contribution": stock_contributions_percent})
    st.write("## Stock Contributions to Portfolio")
    st.table(stock_contributions)
    
    # Fit a linear regression model to the data
    x_vals = np.arange(len(portfolio_returns_monthly))
    y_vals = portfolio_returns_monthly['Portfolio'].values
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    trendline = slope * x_vals + intercept

    fig_portfolio_returns.add_trace(go.Scatter(x=months_abbr, y=trendline, name="Trendline", mode='lines', line=dict(dash='dash')))
    fig_portfolio_returns.update_layout(title_text='Portfolio Monthly Returns - Last 1 Year',
                                        xaxis_title='Date',
                                        yaxis_title='Return',
                                        xaxis_tickvals=list(range(len(months_abbr))),
                                        xaxis_ticktext=months_abbr,
                                        xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_portfolio_returns)
    
    