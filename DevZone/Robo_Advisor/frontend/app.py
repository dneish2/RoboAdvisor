import sys
import os
import re
import json  # Added for JSON parsing

# Add the parent directory to sys.path to locate the backend package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 3 Features
# 1. Suggestion of either a put or option position given the last [timeframe] of the stock's option contract
# 2. Summarize the Monte Carlo simulation and provide a guided or balanced view of the results
# 3. When the name of a stock price is given or changes, produce a chart of the stock price over the past year
# 4. Include BTC, and Housing assets to portfolio optimization based on risk tolerance

import streamlit as st
import pandas as pd
from backend.models import UserProfile
from backend.portfolio_optimizer import PortfolioOptimizer
from backend.data_fetcher import DataFetcher
from backend.llama_integration import LLamaQuery
from backend.options_visualizer import OptionsVisualizer
import plotly.express as px
import pyttsx3
import reticker  # New import for ticker extraction
import yfinance as yf  # New import for stock data
import plotly.graph_objs as go  # New import for modern charts

# Helper function to validate ticker symbols
def is_valid_ticker(ticker):
    return re.match("^[A-Za-z]{1,5}$", ticker) is not None

# Helper function to extract tickers from text
def extract_tickers(text):
    extractor = reticker.TickerExtractor()
    return extractor.extract(text)

# Helper function to fetch historical stock data
def get_historical_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        return None
    return hist

# Helper function to plot stock chart using Plotly
def plot_stock_chart(ticker, data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name=f'{ticker} Close Price',
        line=dict(color='cyan')
    ))
    fig.update_layout(
        title=f"{ticker} - 1 Year Performance",
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        hovermode='x unified'
    )
    st.plotly_chart(fig)

# Cache the LLamaQuery initialization to prevent multiple initializations
@st.cache_resource(show_spinner=False)
def get_llama_query():
    return LLamaQuery(max_documents=10)

# Initialize LLamaIndex using the cached function
try:
    llama_query = get_llama_query()
    st.success("LLamaIndex initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize LLamaIndex: {e}")

# Similarly, cache DataFetcher initialization
@st.cache_resource(show_spinner=False)
def get_data_fetcher():
    return DataFetcher()

try:
    data_fetcher = get_data_fetcher()
    st.success("DataFetcher initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize DataFetcher: {e}")

# Cache the Text-to-Speech engine to avoid repeated initializations
@st.cache_resource(show_spinner=False)
def get_tts_engine():
    try:
        engine = pyttsx3.init()
        return engine
    except Exception as e:
        st.error(f"Failed to initialize Text-to-Speech: {e}")
        return None

tts_engine = get_tts_engine()
if tts_engine:
    st.success("Text-to-Speech initialized successfully.")

def gather_requirements():
    st.title("Robo-Advisor: Gather Your Financial Goals")

    # User Goals
    st.header("Your Financial Goals")
    num_goals = st.number_input("Number of Goals", min_value=1, max_value=10, value=1, step=1)
    goals = []
    for i in range(num_goals):
        st.subheader(f"Goal {i+1}")
        goal = st.text_input(f"Goal {i+1} Description", key=f'goal_{i}')
        amount = st.number_input(f"Amount for Goal {i+1} ($)", min_value=0.0, value=10000.0, key=f'amount_{i}')
        time_horizon = st.number_input(f"Time Horizon for Goal {i+1} (years)", min_value=0.1, max_value=30.0, value=5.0, step=0.1, key=f'horizon_{i}')
        goals.append({'description': goal, 'amount': amount, 'time_horizon': time_horizon})

    # Risk Tolerance
    st.header("Risk Tolerance")
    risk = st.selectbox("Select your risk tolerance level", ["Low", "Medium", "High"])

    # Available Investment Amount
    st.header("Available Investment Amount")
    available_investment = st.number_input("Amount to Invest ($)", min_value=0.0, value=2000.0, step=1000.0)

    if st.button("Submit"):
        user_profile = UserProfile(
            goals=goals,
            risk_tolerance=risk,
            available_investment=available_investment
        )
        user_profile.save_to_csv('data/user_data.csv')
        st.success("Requirements gathered successfully!")
        return user_profile
    return None

def display_portfolio(user_profile):
    if user_profile:
        # Validate 'goals' is a list
        if not isinstance(user_profile.get('goals'), list):
            st.error(f"'goals' should be a list, got {type(user_profile.get('goals'))}")
            return

        optimizer = PortfolioOptimizer(user_profile)
        try:
            portfolio, expected_return, simulation_results = optimizer.optimize()

            # Display Portfolio Allocation
            st.header("Optimal Portfolio Allocation")
            fig_allocation = px.pie(
                portfolio, 
                names='Asset', 
                values='Allocation', 
                title='Portfolio Allocation', 
                hole=0.3, 
                template='plotly_dark'
            )
            st.plotly_chart(fig_allocation)

            # Display Expected Return
            st.subheader(f"Expected Annual Return: {expected_return*100:.2f}%")

            # Monte Carlo Simulation Results
            st.header("Monte Carlo Simulation Results")
            fig_sim = px.histogram(
                simulation_results, 
                nbins=50, 
                title='Monte Carlo Simulation of Portfolio Returns', 
                template='plotly_dark'
            )
            st.plotly_chart(fig_sim)

            # Text-to-Speech for Portfolio Allocation
            if tts_engine and st.button("Read Portfolio Allocation"):
                for index, row in portfolio.iterrows():
                    text = f"{row['Asset']}: {row['Allocation']*100:.2f}%"
                    tts_engine.say(text)
                tts_engine.runAndWait()

            # Portfolio Growth Over Time
            st.header("Portfolio Growth Over Time")
            historical_data = data_fetcher.get_historical_performance(portfolio['Asset'].tolist())

            if historical_data.empty:
                st.warning("No historical data available for the selected assets.")
            else:
                # Calculate cumulative returns
                normalized_prices = historical_data / historical_data.iloc[0]

                # Get allocations
                allocations = portfolio.set_index('Asset')['Allocation']
                allocated_amount = allocations * user_profile['available_investment']

                # Calculate portfolio value over time
                portfolio_values = normalized_prices * allocated_amount
                portfolio_values['Total'] = portfolio_values.sum(axis=1)

                fig_growth = px.line(
                    portfolio_values, 
                    x=portfolio_values.index, 
                    y='Total',
                    title='Total Portfolio Growth Over Time',
                    labels={'x': 'Date', 'Total': 'Portfolio Value ($)'},
                    template='plotly_dark'
                )
                st.plotly_chart(fig_growth)

                # Individual Asset Performance
                fig_individual = px.line(
                    portfolio_values, 
                    x=portfolio_values.index, 
                    y=portfolio['Asset'],
                    title='Individual Asset Performance Over Time',
                    labels={'x': 'Date', 'value': 'Value ($)', 'variable': 'Asset'},
                    template='plotly_dark'
                )
                st.plotly_chart(fig_individual)

            return portfolio
        except Exception as e:
            st.error(f"Portfolio optimization failed: {e}")
    return None

def query_llama():
    st.header("Ask About Companies")
    user_query = st.text_input("Enter your question about companies' reports:")
    if st.button("Ask"):
        if user_query.strip() == "":
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Processing your query..."):
                try:
                    # Query LLamaIndex
                    response = llama_query.query(user_query)
                    st.write(response)

                    # Extract and visualize tickers
                    tickers = extract_tickers(user_query)
                    if tickers:
                        st.subheader("Stock Performance Charts")
                        for ticker in tickers:
                            ticker = ticker.upper()
                            if is_valid_ticker(ticker):
                                data = get_historical_data(ticker)
                                if data is not None:
                                    plot_stock_chart(ticker, data)
                                else:
                                    st.warning(f"No data found for ticker: {ticker}")
                            else:
                                st.warning(f"Invalid ticker symbol: {ticker}")
                    else:
                        st.info("No valid stock tickers found in your query.")

                except Exception as e:
                    st.error(f"An error occurred while processing your query: {e}")

def visualize_options():
    st.header("Options Contracts Visualization")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")
    period = st.selectbox("Select Period", ["1 Week", "1 Month", "3 Months", "6 Months"])

    if st.button("Fetch Options Data"):
        if ticker.strip() == "":
            st.warning("Please enter a valid stock ticker.")
        elif not is_valid_ticker(ticker.upper()):
            st.warning("Please enter a valid stock ticker symbol.")
        else:
            with st.spinner("Fetching options data..."):
                try:
                    options_data = data_fetcher.get_options_data(ticker.upper(), period)
                    if not options_data.empty:
                        # Instantiate OptionsVisualizer for Appreciation Visualization
                        visualizer = OptionsVisualizer(options_data, time_frame=period)
                        fig = visualizer.plot_options_chain()
                        st.plotly_chart(fig)
                    else:
                        st.warning("No options data found for the given ticker and period.")
                except Exception as e:
                    st.error(f"An error occurred while fetching options data: {e}")

def visualize_history():
    st.header("Option Contract History Visualization")
    
    # Get ticker input
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")

    if ticker:
        # Get period selection
        period = st.selectbox(
            "Select Time Period",
            ["1 Week", "1 Month", "3 Months", "6 Months"]
        )
        
        # Fetch options data which includes all contracts and their details
        options_data = data_fetcher.get_options_data(ticker.upper(), period)
        
        if options_data.empty:
            st.warning("No options data available for this ticker and period.")
            return
            
        # Get unique expiration dates from the fetched data
        expiration_dates = sorted(options_data['Expiration'].unique())
        
        if expiration_dates:
            # Select expiration date
            selected_expiration = st.selectbox("Select Expiration Date", expiration_dates)
            
            # Filter options for selected expiration
            expiration_options = options_data[options_data['Expiration'] == selected_expiration].copy()
            
            # Get unique strikes for the selected expiration
            strike_prices = sorted(expiration_options['Strike'].unique())
            
            if strike_prices:
                # Select strike price
                selected_strike = st.selectbox("Select Strike Price", strike_prices)
                
                # Filter and display the options chain for selected strike
                selected_options = expiration_options[expiration_options['Strike'] == selected_strike].copy()
                
                # Show options chain data
                if not selected_options.empty:
                    st.write("Options Chain Data:")
                    st.dataframe(selected_options)
                    
                    # Identify all available option types for the selected strike
                    option_types_available = selected_options['Type'].unique()
                    
                    # Inform the user about available option types
                    if len(option_types_available) > 1:
                        st.info(f"Displaying historical data for **{', '.join(option_types_available)}** options.")
                    else:
                        st.info(f"Displaying historical data for **{option_types_available[0]}** options.")
                    
                    # Fetch and plot historical data for each available option type
                    for option_type in option_types_available:
                        option_history = data_fetcher.get_option_history(
                            ticker=ticker.upper(),
                            expiration=selected_expiration,
                            strike=selected_strike,
                            option_type=option_type
                        )
                        
                        if option_history.empty:
                            st.warning(f"No historical data available for the selected {option_type} option.")
                        else:
                            # Instantiate OptionsVisualizer with no options data
                            visualizer = OptionsVisualizer()
                            fig_history = visualizer.plot_option_history(option_history, option_type)
                            st.plotly_chart(fig_history)
                else:
                    st.warning("No data available for selected strike price.")
            else:
                st.warning("No strike prices available for selected expiration date.")
        else:
            st.warning("No expiration dates available in the selected period.")

def debug_environment():
    st.header("🛠️ Debugging Environment")
    cwd = os.getcwd()
    st.write(f"**Current Working Directory:** `{cwd}`")
    
    dirs_to_list = ['backend', 'storage', 'cache', 'data']
    for directory in dirs_to_list:
        path = os.path.join(cwd, directory)
        st.write(f"**Contents of `{directory}/`:**")
        if os.path.exists(path):
            contents = os.listdir(path)
            if contents:
                st.write(contents)
            else:
                st.write("Directory is empty.")
        else:
            st.write("Directory does not exist.")

def main():
    debug_environment()
    st.sidebar.title("Robo-Advisor")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Gather Requirements", "View Portfolio", "Chat", "Appreciation Visualization", "Visualize History"]
    )

    if app_mode == "Gather Requirements":
        user_profile = gather_requirements()
    elif app_mode == "View Portfolio":
        # Load the last user profile
        try:
            user_profiles = UserProfile.load_from_csv('data/user_data.csv')
            if user_profiles.empty:
                st.error("No user data found. Please gather requirements first.")
                return
            last_profile = user_profiles.iloc[-1]
            
            # Parse the 'goals' JSON string into a Python list
            try:
                goals = json.loads(last_profile['goals'])
                if not isinstance(goals, list):
                    raise ValueError(f"'goals' should be a list, got {type(goals)}")
            except json.JSONDecodeError as e:
                st.error(f"Error parsing 'goals': {e}")
                return
            except ValueError as ve:
                st.error(str(ve))
                return
            
            user_profile = {
                'goals': goals,
                'risk_tolerance': last_profile['risk_tolerance'],
                'available_investment': float(last_profile['available_investment'])
            }
            st.write("Loaded User Profile:", user_profile)
            display_portfolio(user_profile)
        except Exception as e:
            st.error(f"Error loading user data: {e}")
    elif app_mode == "Chat":
        query_llama()
    elif app_mode == "Appreciation Visualization":
        visualize_options()
    elif app_mode == "Visualize History":
        visualize_history()

if __name__ == "__main__":
    main()