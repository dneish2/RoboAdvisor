# frontend/app.py

import sys
import os
import re
import json  # For JSON parsing
import logging

# Add the parent directory to sys.path to locate the backend package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Streamlit and Data Visualization Imports
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Backend Module Imports
from backend.models import UserProfile
from backend.portfolio_optimizer import PortfolioOptimizer
from backend.data_fetcher import DataFetcher
from backend.llama_integration import LLamaQuery
from backend.options_visualizer import OptionsVisualizer

# Additional Imports
import pyttsx3  # For Text-to-Speech
import reticker  # For ticker extraction
import yfinance as yf  # For stock data
from openai import OpenAI  # For NVIDIA NIM Client
from dotenv import load_dotenv  # For environment variables

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Environment Variables
load_dotenv(override=True)

# Initialize NVIDIA NIM Client
api_key = os.getenv("NIM_API_KEY")
if not api_key:
    st.error("API Key not found. Please ensure your .env file contains a valid NIM_API_KEY.")
    st.stop()

nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key  # Use the loaded API key
)

# Helper Functions

def is_valid_ticker(ticker):
    """Validate ticker symbols using regex."""
    return re.match("^[A-Za-z]{1,5}(\.[A-Za-z]{1,2})?$", ticker) is not None

def extract_tickers(text):
    """Extract tickers from user input text."""
    extractor = reticker.TickerExtractor()
    return extractor.extract(text)

def get_historical_data(ticker, period='1y'):
    """Fetch historical stock data using yfinance."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        return None
    return hist

def plot_stock_chart(ticker, data):
    """Plot stock chart using Plotly."""
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

def display_portfolio(user_profile, nim_client, data_fetcher, max_holdings=4):
    if not user_profile:
        st.error("User profile is missing.")
        return

    # Validate and parse 'goals'
    goals = user_profile.get('goals')
    if isinstance(goals, str):
        try:
            goals = json.loads(goals)
            if not isinstance(goals, list):
                raise ValueError(f"'goals' should be a list after parsing, got {type(goals)}")
            user_profile['goals'] = goals  # Update the user_profile with parsed goals
        except json.JSONDecodeError:
            st.error("Invalid JSON format for 'goals'. Please ensure it is properly formatted.")
            return
        except ValueError as ve:
            st.error(str(ve))
            return

    # Load the investment thesis
    investment_thesis = data_fetcher.load_investment_thesis()
    if investment_thesis.empty:
        st.error("Investment thesis could not be loaded.")
        return

    # Initialize PortfolioOptimizer with user profile, investment thesis, and max_holdings
    try:
        optimizer = PortfolioOptimizer(user_profile, investment_thesis, max_holdings=max_holdings)
    except ValueError as ve:
        st.error(f"Portfolio optimization initialization failed: {ve}")
        return
    except KeyError as ke:
        st.error(f"Portfolio optimization initialization failed: {ke}")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during initialization: {e}")
        return

    # Optimize the portfolio
    try:
        portfolio, expected_return, simulation_results = optimizer.optimize()
        # Inform the user about the limited holdings
        if len(optimizer.assets) < max_holdings:
            st.warning(f"Only {len(optimizer.assets)} assets were available and included in the portfolio based on your risk tolerance.")
        # Display the portfolio, expected return, and simulation results
        # Display only the portfolio, hiding other data
        st.write("Optimized Portfolio:")
        st.dataframe(portfolio)
    except ValueError as ve:
        st.error(f"Portfolio optimization failed: {ve}")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during optimization: {e}")
        return

    # Convert portfolio to DataFrame if it's not already
    if not isinstance(portfolio, pd.DataFrame):
        try:
            portfolio_df = pd.DataFrame(portfolio)
        except Exception as e:
            st.error(f"Failed to convert portfolio to DataFrame: {e}")
            return
    else:
        portfolio_df = portfolio.copy()

    # Display Portfolio Allocation
    st.header("Optimal Portfolio Allocation")
    try:
        fig_allocation = px.pie(
            portfolio_df,
            names='Asset',
            values='Allocation',
            title='Portfolio Allocation',
            hole=0.3,
            template='plotly_dark'
        )
        st.plotly_chart(fig_allocation)
    except Exception as e:
        st.error(f"Failed to plot portfolio allocation: {e}")

    # Display Expected Return
    st.subheader(f"Expected Annual Return: {expected_return * 100:.2f}%")

    # Monte Carlo Simulation Results
    st.header("Monte Carlo Simulation Results")
    try:
        simulation_df = pd.DataFrame({'Returns': simulation_results})
        fig_sim = px.histogram(
            simulation_df,
            x='Returns',
            nbins=50,
            title='Monte Carlo Simulation of Portfolio Returns',
            template='plotly_dark'
        )
        st.plotly_chart(fig_sim)
    except Exception as e:
        st.error(f"Failed to plot Monte Carlo simulation results: {e}")

    # Text-to-Speech for Portfolio Allocation
    if tts_engine:
        if st.button("Read Portfolio Allocation"):
            try:
                for _, row in portfolio_df.iterrows():
                    text = f"{row['Asset']}: {row['Allocation'] * 100:.2f}%"
                    tts_engine.say(text)
                tts_engine.runAndWait()
                st.success("Portfolio allocation has been read aloud.")
            except Exception as e:
                st.error(f"Text-to-Speech failed: {e}")

    # Generate Portfolio Summary using NVIDIA NIM API
    if nim_client:
        description_prompt = (
            f"Provide a comprehensive summary of the user's investment portfolio based on the following details:\n\n"
            f"**Portfolio Composition:** {portfolio_df.to_dict(orient='records')}\n"
            f"**Number of Simulations Run:** {len(simulation_results)}\n"
            f"**Simulation Purpose:** To assess the distribution and potential outcomes of portfolio returns.\n"
            f"**Estimated Annual Return:** {expected_return * 100:.2f}%\n"
            f"**Timeframe:** {goals[0].get('time_horizon', 'N/A')} years\n\n"
            f"Provide the summary in a clear and concise manner, suitable for a user without financial expertise. "
            f"Use markdown formatting for better readability, including:\n"
            f"- Headers (##) for main sections\n"
            f"- Bold (**) for key metrics\n"
            f"- Bullet points (-) for lists\n"
            f"- Tables where appropriate\n"
        )

        try:
            completion = nim_client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": description_prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=False
            )

            if completion and completion.choices:
                # Corrected access to the 'content' attribute
                portfolio_description = completion.choices[0].message.content.strip()
                if portfolio_description:
                    st.header("Portfolio Summary")

                    # Add disclaimer to the portfolio description
                    DISCLAIMER_TEXT = (
                        "\n\n---\n\n"  # Add a markdown horizontal line before disclaimer
                        "### Disclaimer\n\n"
                        "Please remember past performance is NOT indicative of future results, but we can use them as a benchmark/scale for the future.\n\n"
                    )

                    # Combine description and disclaimer, then render as markdown
                    full_content = portfolio_description + DISCLAIMER_TEXT

                    # Create a container for better spacing
                    with st.container():
                        st.markdown(full_content)
                else:
                    st.warning("Portfolio summary is empty.")
            else:
                st.warning("Unable to generate portfolio summary at this time.")
        except Exception as e:
            st.warning(f"Could not generate portfolio summary: {e}")

    # Portfolio Growth Over Time
    st.header("Portfolio Growth Over Time")
    try:
        historical_data = data_fetcher.get_historical_performance(portfolio_df['Asset'].tolist())
        if historical_data.empty:
            st.warning("No historical data available for the selected assets.")
        else:
            # Normalize prices to start at 1 for comparison
            normalized_prices = historical_data / historical_data.iloc[0]

            # Calculate allocated amounts
            allocations = portfolio_df.set_index('Asset')['Allocation']
            allocated_amount = allocations * user_profile['available_investment']

            # Calculate portfolio values
            portfolio_values = normalized_prices.mul(allocated_amount, axis=1)
            portfolio_values['Total'] = portfolio_values.sum(axis=1)

            # Plot Total Portfolio Growth
            fig_growth = px.line(
                portfolio_values,
                y='Total',
                title='Total Portfolio Growth Over Time',
                labels={'index': 'Date', 'Total': 'Portfolio Value ($)'},
                template='plotly_dark'
            )
            st.plotly_chart(fig_growth)

            # Plot Individual Asset Performance
            individual_performance = portfolio_values.drop('Total', axis=1).copy()
            fig_individual = px.line(
                individual_performance,
                title='Individual Asset Performance Over Time',
                labels={'index': 'Date', 'value': 'Value ($)', 'variable': 'Asset'},
                template='plotly_dark'
            )
            st.plotly_chart(fig_individual)
    except Exception as e:
        st.error(f"Failed to plot portfolio growth: {e}")

    return portfolio_df

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

def visualize_options(data_fetcher):
    st.header("Instrument Scanner")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")
    period = st.selectbox("Select Period", ["1 Week", "1 Month", "3 Months", "6 Months"])

    options_ready = st.session_state.get("options_data_ready", False)
    open_sim = st.button("Open OptionSim Pro", disabled=not options_ready)

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
                        st.session_state["options_data"] = options_data
                        st.session_state["options_data_ready"] = True
                        st.session_state["options_ticker"] = ticker.upper()
                        st.session_state["options_period"] = period

                        # Instantiate OptionsVisualizer for Appreciation Visualization
                        visualizer = OptionsVisualizer(options_data, time_frame=period)
                        fig = visualizer.plot_options_chain()
                        st.plotly_chart(fig)
                        st.success("Options data fetched. Launch OptionSim Pro to explore contracts interactively.")
                    else:
                        st.session_state["options_data_ready"] = False
                        st.session_state["options_data"] = None
                        st.warning("No options data found for the given ticker and period.")
                except Exception as e:
                    st.session_state["options_data_ready"] = False
                    st.session_state["options_data"] = None
                    st.error(f"An error occurred while fetching options data: {e}")

    if open_sim:
        st.session_state["app_mode"] = "OptionSim Pro"
        st.experimental_rerun()

def visualize_history(data_fetcher, nim_client):
    st.header("Instrument History")

    # Get ticker input
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")

    if ticker:
        # Validate ticker symbol
        if not is_valid_ticker(ticker.upper()):
            st.warning("Please enter a valid stock ticker symbol.")
            return

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

        # Ensure 'Expiration' column is in datetime format
        if options_data['Expiration'].dtype == object:
            try:
                options_data['Expiration'] = pd.to_datetime(options_data['Expiration'])
            except Exception as e:
                st.error(f"Error parsing 'Expiration' dates: {e}")
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

                # Show options chain data without displaying expiration time again
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

                    # Let user select option type if multiple are available
                    if len(option_types_available) > 1:
                        option_type_selected = st.selectbox("Select Option Type", option_types_available)
                    else:
                        option_type_selected = option_types_available[0]

                    # Prepare the expiration date string in 'YYYY-MM-DD' format
                    try:
                        expiration_str = selected_expiration.strftime('%Y-%m-%d')
                    except Exception as e:
                        st.error(f"Error formatting expiration date: {e}")
                        return

                    # Fetch and plot historical data for the selected option type
                    option_history = data_fetcher.get_option_history(
                        ticker=ticker.upper(),
                        expiration=expiration_str,
                        strike=selected_strike,
                        option_type=option_type_selected
                    )

                    if not option_history.empty:
                        # Instantiate OptionsVisualizer with the historical data
                        visualizer = OptionsVisualizer()
                        fig_history = visualizer.plot_option_history(option_history, option_type_selected)
                        st.plotly_chart(fig_history)

                        # **Strategic Options Advisor Integration in Visualization History**
                        st.header("🧠 Strategic Options Advisor for Contract History")
                        st.markdown("### Get Personalized Strategic Advice Based on Contract History")

                        # Use an expandable section to display strategic advice
                        with st.expander("Click to Get Strategic Advice"):
                            # Construct the prompt using selected contract details and historical data
                            historical_closes = option_history['Close'].tolist()

                            advisor_prompt_history = (
                            f"You are a strategic options advisor with extensive knowledge in market trends and option strategies.\n\n"
                            f"Analyze the historical performance of the following options contract and provide strategic advice to optimize entry and exit points for maximizing profits.\n\n"
                            f"**Option Details:**\n"
                            f"- **Ticker:** {ticker.upper()}\n"
                            f"- **Option Type:** {option_type_selected}\n"
                            f"- **Strike Price:** ${selected_strike:.2f}\n"
                            f"- **Expiration Date:** {expiration_str}\n\n"
                            f"**Historical Close Prices:**\n"
                            f"{historical_closes}\n\n"
                            f"Please provide a concise (no more than 300 words), beginner-friendly strategy that includes:\n"
                            f"- **Potential Entry Points:** When and why to enter the trade.\n"
                            f"- **Target Exit Prices:** Clear price levels to sell the option for profits.\n"
                            f"- **Overview of Highs and Lows:** Summarize the highest and lowest option prices during the contract's timeframe.\n"
                            f"- **Associated Risks:** Outline the main risks involved with this strategy.\n\n"
                            f"Use simple language and avoid technical jargon. Format your response using markdown with appropriate headers and bullet points for clarity.")

                            # Button to get strategic advice based on history
                            if st.button("Get Strategic Advice Based on History"):
                                # Call NVIDIA NIM API to get strategic advice
                                try:
                                    advisor_completion_history = nim_client.chat.completions.create(
                                        model="meta/llama-3.1-405b-instruct",
                                        messages=[{"role": "user", "content": advisor_prompt_history}],
                                        temperature=0.3,
                                        top_p=0.7,
                                        max_tokens=1024,
                                        stream=False
                                    )

                                    if advisor_completion_history and advisor_completion_history.choices:
                                        # Corrected access to the 'content' attribute
                                        advisor_response_history = advisor_completion_history.choices[0].message.content.strip()

                                        if advisor_response_history:
                                            st.markdown("### **Strategic Advice Based on History**")
                                            st.markdown(advisor_response_history)
                                        else:
                                            st.error("Failed to generate strategic advice based on history.")
                                    else:
                                        st.error("No response received from the strategic advisor.")
                                except Exception as e:
                                    st.error(f"Error generating strategic advice: {e}")
                    else:
                        st.warning("No historical data available for the selected option type.")
                else:
                    st.warning("No strike prices available for the selected expiration date.")
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
        [
            "Gather Requirements",
            "View Portfolio",
            "Chat",
            "Instrument Scanner",
            "Instrument History",
            "OptionSim Pro",
        ],
        key="app_mode",
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
            display_portfolio(user_profile, nim_client, data_fetcher, max_holdings=4)  # Set max_holdings here
        except Exception as e:
            st.error(f"Error loading user data: {e}")
    elif app_mode == "Chat":
        query_llama()
    elif app_mode == "Instrument Scanner":
        visualize_options(data_fetcher)  # Removed nim_client
    elif app_mode == "Instrument History":
        visualize_history(data_fetcher, nim_client)  # Pass data_fetcher and nim_client to visualize_history
    elif app_mode == "OptionSim Pro":
        try:
            from frontend.options_sim import render_option_sim
        except Exception:
            from options_sim import render_option_sim

        render_option_sim(data_fetcher)
    else:
        st.error("Invalid app mode selected.")

if __name__ == "__main__":
    main()