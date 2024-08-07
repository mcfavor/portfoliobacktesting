import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# Initialize session state for storing tickers, weights, and names
if 'tickers' not in st.session_state:
    st.session_state['tickers'] = []
if 'period' not in st.session_state:
    st.session_state['period'] = '1y'

# Function to add a ticker with weight and full name
def add_ticker(ticker, weight):
    try:
        ticker_data = yf.Ticker(ticker)
        full_name = ticker_data.info.get('longName', 'N/A')
    except Exception as e:
        full_name = 'N/A'
        st.error(f"Error fetching data for {ticker}: {e}")
    
    if ticker not in [t['Ticker'] for t in st.session_state['tickers']]:
        st.session_state['tickers'].append({'Ticker': ticker, 'Weight': weight, 'Full Name': full_name})

# Function to remove a ticker
def remove_ticker(ticker):
    st.session_state['tickers'] = [t for t in st.session_state['tickers'] if t['Ticker'] != ticker]

# Function to check if weights sum up to 100
def check_weights():
    total_weight = sum(t['Weight'] for t in st.session_state['tickers'])
    return total_weight

# Function to calculate portfolio statistics
def calculate_portfolio_statistics(period, risk_free_rate):
    tickers = [t['Ticker'] for t in st.session_state['tickers']]
    weights = [t['Weight'] / 100 for t in st.session_state['tickers']]
    data = yf.download(tickers, period=period)['Adj Close']
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()
    port_variance = np.dot(weights, np.dot(cov_matrix, weights))
    port_std_dev = np.sqrt(port_variance)

    annualized_returns = (1 + returns.mean()) ** 252 - 1  # Assuming 252 trading days in a year
    port_annualized_return = np.dot(weights, annualized_returns)

    # Calculate Sharpe Ratio
    sharpe_ratio = (port_annualized_return - risk_free_rate) / port_std_dev

    return port_variance, port_std_dev, port_annualized_return, sharpe_ratio

# Define the function to fetch data and calculate portfolio value
def calculate_portfolio_value(initial_investment, period):
    tickers = [t['Ticker'] for t in st.session_state['tickers']]
    weights = {t['Ticker']: t['Weight'] / 100 for t in st.session_state['tickers']}

    # Download historical data
    data = yf.download(tickers, period=period)['Adj Close']

    # Calculate daily returns
    daily_returns = data.pct_change().dropna()

    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod()

    # Calculate portfolio cumulative returns
    portfolio_cumulative_returns = (cumulative_returns * pd.Series(weights)).sum(axis=1)

    # Calculate the portfolio value over time
    portfolio_value = initial_investment * portfolio_cumulative_returns

    return portfolio_value

# Function to calculate benchmark value
def calculate_benchmark_value(initial_investment, benchmark_ticker, period):
    # Download historical data
    benchmark_data = yf.download(benchmark_ticker, period=period)['Adj Close']

    # Calculate daily returns
    benchmark_daily_returns = benchmark_data.pct_change().dropna()

    # Calculate cumulative returns
    benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod()

    # Calculate the benchmark value over time
    benchmark_value = initial_investment * benchmark_cumulative_returns

    return benchmark_value

# Function to calculate annual portfolio returns
def calculate_annual_returns(value_series):
    annual_returns = value_series.resample('Y').last().pct_change().dropna()
    return annual_returns

# Streamlit app
st.write("""
# Backtest Portfolio Asset Class Allocation

##### This portfolio backtesting tool allows you to construct one portfolio based on the selected mutual funds, ETFs, and stocks. You can analyze and backtest portfolio return and risk characteristics.

""")

# Dropdown for selecting period
period = st.sidebar.selectbox(
    "Select the period for historical data:",
    ('1y', '2y', '5y', '10y', 'ytd', 'max'),
    key='period'
)

# Input for risk-free rate
risk_free_rate = st.sidebar.number_input("Risk-free rate (as a decimal):", value=0.03)
initial_investment = st.sidebar.number_input('Initial Investment ($)', value=10000)

benchmark_ticker = st.sidebar.text_input("Add the Benchmark ticker symbol:")

# Input for adding a new ticker
new_ticker = st.sidebar.text_input("Add the Asset ticker symbol:")
new_weight = st.sidebar.number_input("Add the Asset weight:", max_value=100)

if st.sidebar.button("Add Ticker"):
    if new_ticker:
        add_ticker(new_ticker, new_weight)

# Display added tickers in a table
st.write("### Added Financial Instruments:")
if st.session_state['tickers']:
    tickers_df = pd.DataFrame(st.session_state['tickers'])
    tickers_df.loc['Total'] = tickers_df.sum(numeric_only=True)
    tickers_df.at['Total', 'Ticker'] = 'Total'
    tickers_df.at['Total', 'Full Name'] = ''
    st.table(tickers_df)

    total_weight = check_weights()
    if total_weight != 100:
        st.warning(f"Total asset weights must sum up to 100. Current total: {total_weight}")

    # Select ticker to remove
    ticker_to_remove = st.sidebar.selectbox("Select a ticker to remove:", [t['Ticker'] for t in st.session_state['tickers']])
    if st.sidebar.button("Remove Ticker"):
        if ticker_to_remove:
            remove_ticker(ticker_to_remove)

    # Assuming tickers_df is your DataFrame
    weights = tickers_df[:-1]['Weight'].astype(float)  # Exclude the total row
    labels = tickers_df[:-1]['Ticker']

    st.write("### Asset Allocation:")

    # Create a pie chart using Plotly
    fig = px.pie(values=weights, names=labels, hole=0.3)

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    # Calculate and display portfolio statistics
    port_variance, port_std_dev, port_annualized_return, sharpe_ratio = calculate_portfolio_statistics(st.session_state['period'], risk_free_rate)

    # Calculate portfolio value
    if st.button('Run Backtest Analysis'):
        with st.spinner('Running Backtest Analysis'):
            portfolio_value = calculate_portfolio_value(initial_investment, period)
            benchmark_value = calculate_benchmark_value(initial_investment, benchmark_ticker, period)

            # Calculate annual returns for portfolio and benchmark
            annual_portfolio_returns = calculate_annual_returns(portfolio_value)
            annual_benchmark_returns = calculate_annual_returns(benchmark_value)

            # Combine annual returns into a single DataFrame
            annual_returns_df = pd.DataFrame({
                'Portfolio': annual_portfolio_returns,
                'Benchmark': annual_benchmark_returns
            }).reset_index()

            # Plot the portfolio and benchmark value over time using st.line_chart
            combined_values = pd.DataFrame({
                'Portfolio': portfolio_value,
                'Benchmark': benchmark_value
            })

        # Display the final portfolio value
        st.write(f'Final Portfolio Value: ${portfolio_value[-1]:.2f}')
        st.write(f'Final Benchmark Value: ${benchmark_value[-1]:.2f}')

        st.write("### Portfolio Growth vs Benchmark Growth")

        st.line_chart(combined_values, x_label="Year", y_label="Amount ($)")

        # Plot the annual returns using Plotly to create a grouped bar chart
        st.write("### Portfolio Annual Returns vs Benchmark Annual Returns")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=annual_returns_df['Date'],
            y=annual_returns_df['Portfolio'],
            name='Portfolio'
        ))
        fig.add_trace(go.Bar(
            x=annual_returns_df['Date'],
            y=annual_returns_df['Benchmark'],
            name='Benchmark'
        ))

        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Return',
            barmode='group'
        )

        st.plotly_chart(fig)

        st.write("### Portfolio Statistics:")
        stats_df = pd.DataFrame({
            'Metric': ['Variance', 'Standard Deviation', 'Annualized Return', 'Sharpe Ratio'],
            'Value': [port_variance, port_std_dev, port_annualized_return, sharpe_ratio]
        })
        st.table(stats_df)
else:
    st.write("No tickers added yet.")
