# Importing necessary libraries
import streamlit as st              # Streamlit for creating the web app interface
import yfinance as yf               # yfinance for fetching financial data
import pandas as pd                 # pandas for data manipulation and analysis
import plotly.express as px         # plotly.express for simple visualizations
import numpy as np                  # numpy for numerical operations
import plotly.graph_objects as go   # plotly.graph_objects for complex visualizations
import time                         # time for adding delays in case of retries

# Initialize session state for storing tickers, weights, and names if they don't already exist
if 'tickers' not in st.session_state:
    st.session_state['tickers'] = []

# Initialize session state for storing tickers, weights, and names if they don't already exist
if 'benchmark' not in st.session_state:
    st.session_state['benchmark'] = []

# Initialize session state for storing the period with a default of 1 year
if 'period' not in st.session_state:
    st.session_state['period'] = '1y'

# Function to add a ticker along with its weight and full name for a certain period dynamically changed by the period in the session
def add_ticker(ticker, weight, period):
    retries = 3  # Number of retries for fetching data in case of errors when fetching the data from the yfinance library
    for attempt in range(retries):
        try:
            data = yf.download(ticker, period=period)  # Download historical data for the ticker
            if not data.empty:  # Check if data is available
                ticker_data = yf.Ticker(ticker)  # Fetch detailed ticker information
                full_name = ticker_data.info['longName']  # Get the full name of the financial instrument
                asset_class = ticker_data.info['quoteType'] # Extract the asset class
                break  # Exit the loop if data is successfully fetched
            else:
                full_name = 'N/A'  # Set full name as 'N/A' if no data is available
                st.error(f"Error fetching data for {ticker}: No historical data available")
                return  # Exit the function if no data is available
        except Exception as e:  # Catch any exceptions during data fetching
            if attempt < retries - 1:  # Retry if not the last attempt
                time.sleep(2)  # Wait for 2 seconds before retrying
                continue
            else:
                full_name = 'N/A'  # Set full name as 'N/A' if all retries fail
                st.error(f"Error fetching data for {ticker}: {e}")
                return  # Exit the function if all retries fail
    
    # Add the ticker to the session state if it's not already present
    if ticker not in [t['Ticker'] for t in st.session_state['tickers']]:
        st.session_state['tickers'].append({'Ticker': ticker, 'Weight': weight, 'Full Name': full_name, 'Asset Class': asset_class})


# Function to add a benchmark ticker along with its weight and full name for a certain period dynamically changed by the period in the session
def add_benchmark(benchmark_ticker, period):
    retries = 3  # Number of retries for fetching data in case of errors when fetching the data from the yfinance library
    for attempt in range(retries):
        try:
            data = yf.download(benchmark_ticker, period=period)  # Download historical data for the ticker
            if not data.empty:  # Check if data is available
                ticker_data = yf.Ticker(benchmark_ticker)  # Fetch detailed ticker information
                full_name = ticker_data.info['longName']  # Get the full name of the financial instrument
                asset_class = ticker_data.info['quoteType'] # Extract the asset class
                break  # Exit the loop if data is successfully fetched
            else:
                full_name = 'N/A'  # Set full name as 'N/A' if no data is available
                st.error(f"Error fetching data for {benchmark_ticker}: No historical data available")
                return  # Exit the function if no data is available
        except Exception as e:  # Catch any exceptions during data fetching
            if attempt < retries - 1:  # Retry if not the last attempt
                time.sleep(2)  # Wait for 2 seconds before retrying
                continue
            else:
                full_name = 'N/A'  # Set full name as 'N/A' if all retries fail
                st.error(f"Error fetching data for {benchmark_ticker}: {e}")
                return  # Exit the function if all retries fail
            
    if benchmark_ticker not in [t['benchmark'] for t in st.session_state['tickers']]:
        st.session_state['benchmark'].append({'Benchmark Ticker': benchmark_ticker, 'Full Name': full_name, 'Asset Class': asset_class})

# Function to remove a ticker from the session state
def remove_ticker(ticker):
    st.session_state['tickers'] = [t for t in st.session_state['tickers'] if t['Ticker'] != ticker]

# Function to check if the total weights of the tickers sum up to 100%
def check_weights():
    total_weight = sum(t['Weight'] for t in st.session_state['tickers'])  # Sum all weights
    return total_weight

# Function to calculate portfolio statistics
def calculate_portfolio_statistics(period, risk_free_rate):
    tickers = [t['Ticker'] for t in st.session_state['tickers']]  # Get the list of tickers
    weights = [t['Weight'] / 100 for t in st.session_state['tickers']]  # Convert weights to decimal
    data = yf.download(tickers, period=period)['Adj Close']  # Download adjusted closing prices for the tickers
    returns = data.pct_change().dropna()  # Calculate daily returns and drop missing values
    cov_matrix = returns.cov()  # Calculate the covariance matrix of returns
    port_variance = np.dot(weights, np.dot(cov_matrix, weights))  # Calculate portfolio variance
    port_std_dev = np.sqrt(port_variance)  # Calculate portfolio standard deviation

    annualized_returns = (1 + returns.mean()) ** 252 - 1  # Calculate annualized returns assuming 252 trading days
    port_annualized_return = np.dot(weights, annualized_returns)  # Calculate portfolio annualized return

    # Calculate Sharpe Ratio (return-to-risk ratio)
    sharpe_ratio = (port_annualized_return - risk_free_rate) / port_std_dev

    return port_variance, port_std_dev, port_annualized_return, sharpe_ratio

# Function to fetch data and calculate portfolio value over time
def calculate_portfolio_value(initial_investment, period):
    tickers = [t['Ticker'] for t in st.session_state['tickers']]  # Get the list of tickers
    weights = {t['Ticker']: t['Weight'] / 100 for t in st.session_state['tickers']}  # Convert weights to decimal

    # Download historical adjusted closing prices for the tickers
    data = yf.download(tickers, period=period)['Adj Close']

    # Calculate daily returns
    daily_returns = data.pct_change().dropna()

    # Calculate cumulative returns over time
    cumulative_returns = (1 + daily_returns).cumprod()

    # Calculate portfolio cumulative returns based on weights
    portfolio_cumulative_returns = (cumulative_returns * pd.Series(weights)).sum(axis=1)

    # Calculate the portfolio value over time based on initial investment
    portfolio_value = initial_investment * portfolio_cumulative_returns

    return portfolio_value

# Function to calculate benchmark value over time
def calculate_benchmark_value(initial_investment, benchmark_ticker, period):
    # Download historical adjusted closing prices for the benchmark ticker
    benchmark_data = yf.download(benchmark_ticker, period=period)['Adj Close']

    # Calculate daily returns for the benchmark
    benchmark_daily_returns = benchmark_data.pct_change().dropna()

    # Calculate cumulative returns for the benchmark
    benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod()

    # Calculate the benchmark value over time based on initial investment
    benchmark_value = initial_investment * benchmark_cumulative_returns

    return benchmark_value

# Function to calculate annual returns from a series of portfolio values
def calculate_annual_returns(value_series):
    annual_returns = value_series.resample('Y').last().pct_change().dropna()  # Resample to yearly frequency and calculate returns
    return annual_returns

# Streamlit app interface
st.write("""
# Backtest Portfolio Asset Class Allocation

##### This portfolio backtesting tool allows you to construct one portfolio based on the selected mutual funds, ETFs, and stocks. You can analyze and backtest portfolio return and risk characteristics.
""")

# Sidebar dropdown for selecting the period of historical data
period = st.sidebar.selectbox(
    "Select the period for historical data:",
    ('1y', '2y', '5y', '10y', 'ytd', 'max'),
    key='period'  # Store the selected period in session state
)

# Sidebar input for risk-free rate
risk_free_rate = st.sidebar.number_input("Risk-free rate (as a decimal):", value=0.05)

# Sidebar input for initial investment amount
initial_investment = st.sidebar.number_input('Initial Investment ($)', value=10000)

# Sidebar input for adding a benchmark ticker
benchmark_ticker = st.sidebar.text_input("Add the Benchmark ticker symbol:")

if st.sidebar.button("Add Benchmark"):
    if benchmark_ticker:
        add_benchmark(benchmark_ticker, st.session_state['period'])


#Display the added benchmark in a table
st.write("### Added Benchmark:")
if st.session_state['benchmark']:
    benchmark_tickers_df = pd.DataFrame(st.session_state['benchmark'])
    st.table(benchmark_tickers_df)

# Sidebar input for adding a new ticker symbol
new_ticker = st.sidebar.text_input("Add the Asset ticker symbol:")

# Sidebar input for adding the weight of the new ticker
new_weight = st.sidebar.number_input("Add the Asset weight:", max_value=100)

# Button to add the ticker to the portfolio
if st.sidebar.button("Add Ticker"):
    if new_ticker:  # Ensure a ticker symbol is provided
        add_ticker(new_ticker, new_weight, st.session_state['period'])  # Add ticker to the portfolio

# Display the added tickers in a table
st.write("### Added Financial Instruments:")
if st.session_state['tickers']:  # Check if there are any tickers added
    tickers_df = pd.DataFrame(st.session_state['tickers'])  # Convert tickers to a DataFrame
    tickers_df.loc['Total'] = tickers_df.sum(numeric_only=True)  # Add a row for the total weights
    tickers_df.at['Total', 'Ticker'] = 'Total'  # Label the total row
    tickers_df.at['Total', 'Full Name'] = ''  # Leave the full name empty for the total row
    st.table(tickers_df)  # Display the table in the Streamlit app

    total_weight = check_weights()  # Check if the total weights sum up to 100%
    if total_weight != 100:  # Display a warning if they don't
        st.warning(f"Total asset weights must sum up to 100. Current total: {total_weight}")

    # Sidebar dropdown to select a ticker to remove
    ticker_to_remove = st.sidebar.selectbox("Select a ticker to remove:", [t['Ticker'] for t in st.session_state['tickers']])
    if st.sidebar.button("Remove Ticker"):  # Button to remove the selected ticker
        if ticker_to_remove:
            remove_ticker(ticker_to_remove)  # Remove the selected ticker

    # Prepare data for a pie chart of asset allocation
    weights = tickers_df[:-1]['Weight'].astype(float)  # Exclude the total row and convert weights to float
    labels = tickers_df[:-1]['Ticker']  # Exclude the total row and get the ticker symbols

    

    # Button to run the backtest analysis
    if st.button('Run Backtest Analysis'):
        with st.spinner('Running Backtest Analysis'):  # Display a spinner while the analysis is running

            st.write("### Asset Allocation:")

            # Create a pie chart using Plotly
            fig = px.pie(values=weights, names=labels, hole=0.3)

            # Display the pie chart in Streamlit
            st.plotly_chart(fig)

            # Calculate and display portfolio statistics
            port_variance, port_std_dev, port_annualized_return, sharpe_ratio = calculate_portfolio_statistics(st.session_state['period'], risk_free_rate)
            portfolio_value = calculate_portfolio_value(initial_investment, period)  # Calculate portfolio value over time
            benchmark_value = calculate_benchmark_value(initial_investment, benchmark_ticker, period)  # Calculate benchmark value over time

            # Calculate annual returns for both portfolio and benchmark
            annual_portfolio_returns = calculate_annual_returns(portfolio_value)
            annual_benchmark_returns = calculate_annual_returns(benchmark_value)

            # Combine annual returns into a single DataFrame
            annual_returns_df = pd.DataFrame({
                'Portfolio': annual_portfolio_returns,
                'Benchmark': annual_benchmark_returns
            }).reset_index()

            # Combine portfolio and benchmark values into a DataFrame for plotting
            combined_values = pd.DataFrame({
                'Portfolio': portfolio_value,
                'Benchmark': benchmark_value
            })

        # Display the final portfolio value
        st.write(f'Final Portfolio Value: ${portfolio_value[-1]:.2f}')
        st.write(f'Final Benchmark Value: ${benchmark_value[-1]:.2f}')

        st.write("### Portfolio Growth vs Benchmark Growth")

        # Plot the portfolio and benchmark growth over time
        st.line_chart(combined_values, x_label="Year", y_label="Amount ($)")

        # Plot the annual returns using a grouped bar chart
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
            barmode='group'  # Group bars together
        )

        st.plotly_chart(fig)

        st.write("### Portfolio Statistics:")
        # Display portfolio statistics in a table
        stats_df = pd.DataFrame({
            'Metric': ['Variance', 'Standard Deviation', 'Annualized Return', 'Sharpe Ratio'],
            'Value': [port_variance, port_std_dev, port_annualized_return, sharpe_ratio]
        })
        st.table(stats_df)
else:
    st.write("No tickers added yet.")  # Display message if no tickers are added
