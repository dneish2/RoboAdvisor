# backend/data_fetcher.py

import yfinance as yf
import pandas as pd
import logging
# import pandas_datareader as pdr
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        pass

    def load_investment_thesis(self, filepath='data/investment_thesis.csv'):
        """Load investment thesis from a CSV file."""
        try:
            thesis_df = pd.read_csv(filepath)
            if thesis_df.empty:
                logger.warning("Investment thesis file is empty.")
                return pd.DataFrame()
            return thesis_df
        except Exception as e:
            logger.error(f"Error loading investment thesis: {e}")
            return pd.DataFrame()

    def get_historical_performance(self, assets, time_horizon='5y'):
        """Fetch historical data for the given assets based on the specified time horizon."""
        logger.info(f"Fetching historical performance data for assets: {assets} with time horizon: {time_horizon}")
        data = {}
        for asset in assets:
            if asset == 'CSUSHPINSA':
                ticker = '^CSUSHPINSA'
            else:
                ticker = asset
            stock = yf.Ticker(ticker)
            try:
                hist = stock.history(period=time_horizon)
                if hist.empty:
                    logger.warning(f"No historical data available for asset {asset}.")
                    continue
                data[asset] = hist['Close'].copy()
            except Exception as e:
                logger.error(f"Error fetching historical data for {asset}: {e}")
                continue
        if not data:
            logger.error("No historical data found for provided assets.")
            return pd.DataFrame()  # Return an empty DataFrame if no data
        return pd.DataFrame(data).dropna()

    def get_options_data(self, ticker_symbol, period):
        """Fetch options data for a given ticker within a specified period."""
        try:
            ticker = yf.Ticker(ticker_symbol)
            options_dates = ticker.options
            if not options_dates:
                logger.warning(f"No options dates found for {ticker_symbol}")
                return pd.DataFrame()

            period_mapping = {
                '1 Week': timedelta(weeks=1),
                '1 Month': timedelta(days=30),
                '3 Months': timedelta(days=90),
                '6 Months': timedelta(days=180)
            }
            end_date = datetime.now() + period_mapping.get(period, timedelta(days=30))
            valid_dates = [date for date in options_dates if datetime.strptime(date, '%Y-%m-%d') <= end_date]

            if not valid_dates:
                logger.warning(f"No options found within the specified period for {ticker_symbol}")
                return pd.DataFrame()

            all_options = []
            for date in valid_dates:
                try:
                    options_chain = ticker.option_chain(date)
                    calls = options_chain.calls.copy()
                    calls['Type'] = 'CALL'
                    calls['Expiration'] = date
                    puts = options_chain.puts.copy()
                    puts['Type'] = 'PUT'
                    puts['Expiration'] = date
                    all_options.extend([calls, puts])
                except Exception as e:
                    logger.error(f"Error processing options chain for date {date}: {e}")
                    continue

            if not all_options:
                logger.warning("No valid options data collected")
                return pd.DataFrame()

            options_df = pd.concat(all_options, ignore_index=True).rename(columns={
                'strike': 'Strike',
                'volume': 'Volume',
                'openInterest': 'Open Interest',
                'lastPrice': 'Last Price',
                'Expiration': 'Expiration',
                'Type': 'Type'
            })
            return options_df[['Strike', 'Expiration', 'Type', 'Volume', 'Open Interest', 'Last Price']].copy()

        except Exception as e:
            logger.error(f"Error fetching options data for {ticker_symbol}: {e}")
            return pd.DataFrame()

    def get_option_history(self, ticker, expiration, strike, option_type):
        """
        Fetch historical price data for a specific option contract.

        Parameters:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            expiration (str): Expiration date in 'YYYY-MM-DD' format
            strike (float): Strike price of the option
            option_type (str): 'CALL' or 'PUT'

        Returns:
            DataFrame: Historical data with 'Close' column
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            # Fetch the option chain for the specified expiration date
            options_chain = ticker_obj.option_chain(expiration)
            if option_type.upper() == 'CALL':
                option = options_chain.calls.copy()
            elif option_type.upper() == 'PUT':
                option = options_chain.puts.copy()
            else:
                logger.error("Invalid option type. Must be 'CALL' or 'PUT'.")
                return pd.DataFrame()

            # Filter for the specific strike price
            option = option[option['strike'] == strike].copy()
            if option.empty:
                logger.warning("No option found with the specified strike price.")
                return pd.DataFrame()

            # Extract the option symbol
            option_symbol = option['contractSymbol'].iloc[0]

            # Fetch historical data for the option
            option_history = yf.Ticker(option_symbol).history(period="max")

            if option_history.empty:
                logger.warning("No historical data found for the option contract.")
                return pd.DataFrame()

            return option_history[['Close']].copy()

        except Exception as e:
            logger.error(f"Error fetching option history: {e}")
            return pd.DataFrame()

    def get_expiration_dates(self, ticker_symbol):
        """Fetch available expiration dates for a given ticker."""
        try:
            ticker = yf.Ticker(ticker_symbol)
            return ticker.options or []
        except Exception as e:
            logger.error(f"Error fetching expiration dates for {ticker_symbol}: {e}")
            return []

    def get_strikes_for_expiration(self, ticker_symbol, expiration_date):
        """Fetch strike prices for a specific expiration date."""
        try:
            ticker = yf.Ticker(ticker_symbol)
            options_chain = ticker.option_chain(expiration_date)
            if options_chain.calls.empty and options_chain.puts.empty:
                logger.warning(f"No strike data available for {ticker_symbol} on {expiration_date}.")
                return []
            return sorted(set(options_chain.calls['strike'].tolist() + options_chain.puts['strike'].tolist()))
        except Exception as e:
            logger.error(f"Error fetching strike prices for {ticker_symbol} on {expiration_date}: {e}")
            return []

    def get_current_mortgage_rates(self):
        """
        Fetch the current 10-Year Treasury yield and estimate mortgage rates.
        Returns:
            float: Estimated current mortgage interest rate.
        """
        try:
            # Fetch the current 10-Year Treasury yield using yfinance
            treasury_data = yf.Ticker("^TNX")
            current_data = treasury_data.history(period="1d")
            current_yield = current_data['Close'].iloc[-1]
            # TNX data is in percentages; divide by 100 to get decimal form
            current_yield_decimal = current_yield / 100

            # Estimate mortgage rate by adding a spread (e.g., 1.5%)
            mortgage_rate = current_yield_decimal + 0.016  # 1.5% spread
            return mortgage_rate * 100  # Convert back to percentage
        except Exception as e:
            print(f"Error fetching current mortgage rates: {e}")
            return None

    def get_mortgage_rates_history(self):
        """
        Fetch historical 10-Year Treasury yields and estimate mortgage rates.
        Returns:
            DataFrame: DataFrame containing dates and estimated mortgage rates.
        """
        try:
            # Fetch historical 10-Year Treasury yields for the past year
            treasury_data = yf.Ticker("^TNX")
            history = treasury_data.history(period="1y")
            # TNX data is in percentages; divide by 100 to get decimal form
            history['Yield'] = history['Close'] / 100

            # Estimate mortgage rates by adding a spread
            history['Mortgage Rate'] = (history['Yield'] + 0.015) * 100  # Convert to percentage

            # Return the DataFrame with 'Mortgage Rate'
            return history[['Mortgage Rate']]
        except Exception as e:
            print(f"Error fetching mortgage rates history: {e}")
            return pd.DataFrame()