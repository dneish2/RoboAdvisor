# backend/portfolio_optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, user_profile, investment_thesis, max_holdings=5):
        """
        Initialize the PortfolioOptimizer with user profile, investment thesis, and maximum holdings.

        :param user_profile: Dictionary containing 'goals', 'risk_tolerance', 'available_investment'
        :param investment_thesis: Pandas DataFrame containing 'Ticker', 'Theme', 'RiskLevel'
        :param max_holdings: Integer specifying the maximum number of assets in the portfolio
        """
        # Validate user_profile keys
        required_keys = ['goals', 'risk_tolerance', 'available_investment']
        for key in required_keys:
            if key not in user_profile:
                raise KeyError(f"Missing key in user_profile: '{key}'")

        # Validate 'goals' structure
        if not isinstance(user_profile['goals'], list):
            raise TypeError(f"'goals' should be a list, got {type(user_profile['goals'])}")

        self.user_profile = user_profile
        self.investment_thesis = investment_thesis
        self.risk_tolerance = user_profile.get('risk_tolerance', 'Medium')
        self.available_investment = user_profile.get('available_investment', 10000)
        self.max_holdings = max_holdings

        # Select assets based on risk tolerance
        self.assets = self.select_assets_based_on_risk()
        if not self.assets:
            raise ValueError("No assets match the selected risk tolerance.")

        # Limit the number of assets to prevent data alignment issues
        self.assets = self.assets[:self.max_holdings]  # Limit to max_holdings
        logger.info(f"Limiting to top {len(self.assets)} assets for data alignment.")

        # Get user's time horizon and fetch asset data
        self.time_horizon_years = self.get_time_horizon()
        logger.info(f"User's time horizon: {self.time_horizon_years} years")
        self.asset_data = self.fetch_asset_data()
        self.expected_returns = self.calculate_expected_returns()
        self.cov_matrix = self.asset_data.pct_change().dropna().cov().values

    def get_time_horizon(self):
        """
        Determine the time horizon based on user's goals.
        Uses rounding to avoid floating-point precision issues.

        :return: Integer representing the maximum time horizon in years
        """
        return max(int(round(goal['time_horizon'])) for goal in self.user_profile['goals'])

    def select_assets_based_on_risk(self):
        """
        Select assets from the investment thesis based on risk tolerance.

        :return: List of ticker symbols matching the risk tolerance
        """
        # Define risk levels based on user's risk tolerance
        if self.risk_tolerance == 'Low':
            allowed_risks = ['Low']
        elif self.risk_tolerance == 'Medium':
            allowed_risks = ['Low', 'Medium']
        elif self.risk_tolerance == 'High':
            allowed_risks = ['Low', 'Medium', 'High']
        else:
            allowed_risks = ['Low', 'Medium']

        # Filter assets based on allowed risk levels
        selected_assets = self.investment_thesis[self.investment_thesis['RiskLevel'].isin(allowed_risks)]

        # Select the first 'max_holdings' assets
        selected_tickers = selected_assets['Ticker'].tolist()

        if len(selected_tickers) > self.max_holdings:
            selected_tickers = selected_tickers[:self.max_holdings]
            logger.info(f"Selected top {self.max_holdings} assets based on risk tolerance '{self.risk_tolerance}': {selected_tickers}")
        else:
            logger.info(f"Selected assets based on risk tolerance '{self.risk_tolerance}': {selected_tickers}")

        if not selected_tickers:
            logger.warning("No assets found matching the selected risk tolerance.")

        return selected_tickers

    def optimize(self):
        """
        Optimize the portfolio based on selected assets.

        :return: Tuple containing portfolio DataFrame, expected return, and simulation results
        """
        if len(self.assets) == 0:
            raise ValueError("No valid assets available for optimization.")
        elif len(self.assets) == 1:
            logger.warning("Only one valid asset available. Portfolio optimization requires at least two assets.")
            # Allocate 100% to the single asset
            allocations = [1.0]
            portfolio = pd.DataFrame({
                'Asset': self.assets,
                'Allocation': allocations
            })
            simulation_results = self.run_monte_carlo(allocations)
            expected_return = np.mean(simulation_results)
            portfolio['Investment'] = portfolio['Allocation'] * self.available_investment
            logger.info(f"Portfolio allocation (single asset): {portfolio}")
            return portfolio, expected_return, simulation_results

        num_assets = len(self.assets)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1.0 / num_assets]

        # Objective: Maximize Sharpe Ratio
        def negative_sharpe_ratio(weights):
            ret = self.portfolio_return(weights)
            vol = self.portfolio_volatility(weights)
            sharpe = ret / vol if vol != 0 else 0
            return -sharpe

        result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            allocations = result.x
            portfolio = pd.DataFrame({
                'Asset': self.assets,
                'Allocation': allocations
            })

            # Monte Carlo Simulation for expected returns
            simulation_results = self.run_monte_carlo(allocations)
            expected_return = np.mean(simulation_results)

            # Calculate final portfolio value based on available investment
            portfolio['Investment'] = portfolio['Allocation'] * self.available_investment

            logger.info(f"Optimization successful. Portfolio: {portfolio}")
            return portfolio, expected_return, simulation_results
        else:
            logger.error("Optimization failed.")
            raise Exception("Optimization failed.")

    def fetch_asset_data(self):
        """
        Fetch historical closing price data for selected assets.

        :return: Pandas DataFrame containing aligned historical data
        """
        data = {}
        valid_assets = []
        period_mapping = {1: "1y", 2: "2y", 3: "3y", 5: "5y", 10: "10y"}
        # Ensure time_horizon_years is rounded and converted to int
        period = period_mapping.get(int(round(self.time_horizon_years)), "5y")
        logger.info(f"Fetching historical data with period: '{period}'")

        # Define a mapping for special tickers that require specific formatting
        special_tickers = {
            "BRK.B": "BRK-B",
            # Add more special tickers as needed, excluding problematic foreign tickers
        }

        for asset in self.assets:
            # Use the special ticker if it exists, else use the original ticker
            ticker = special_tickers.get(asset, asset)
            logger.info(f"Processing asset: {asset} (Ticker: {ticker})")

            stock = yf.Ticker(ticker)

            # Attempt to fetch historical data with the specified period
            try:
                hist = stock.history(period=period)
                # Fallback to 1-year if data is unavailable for the requested period
                if hist.empty:
                    logger.warning(f"No historical data for {asset} with period '{period}'. Trying '1y'.")
                    hist = stock.history(period="1y")

                if hist.empty:
                    logger.warning(f"No historical data for {asset}. Skipping.")
                    continue

                data[asset] = hist['Close']
                valid_assets.append(asset)  # Only add asset if data is successfully fetched
                logger.info(f"Successfully fetched data for {asset}")

            except Exception as e:
                logger.error(f"Error fetching historical data for {asset}: {e}")
                continue

        if not data:
            raise ValueError("No historical data found for the selected assets.")

        self.assets = valid_assets  # Update self.assets to include only valid assets
        logger.info(f"Valid assets after fetching data: {self.assets}")

        # Concatenate data using an inner join to retain only common dates
        df = pd.concat(data, axis=1, join='inner')

        if df.empty:
            logger.warning("Asset data is empty after aligning dates.")
            raise ValueError("Asset data is empty after aligning dates. Cannot calculate expected returns.")
        elif df.shape[0] < 100:  # Example threshold
            logger.warning(f"Insufficient data points ({df.shape[0]}) for accurate optimization.")
            raise ValueError("Insufficient historical data for portfolio optimization.")

        logger.info(f"Asset data successfully fetched with shape: {df.shape}")
        return df

    def calculate_expected_returns(self):
        """
        Calculate historical mean returns and adjust based on risk tolerance.

        :return: Numpy array of expected returns
        """
        if self.asset_data.empty:
            raise ValueError("Asset data is empty. Cannot calculate expected returns.")

        mean_returns = self.asset_data.pct_change().dropna().mean().values
        logger.info(f"Calculated mean returns: {mean_returns}")

        # Adjust expected returns based on risk tolerance
        if self.risk_tolerance == 'Low':
            adjusted_returns = mean_returns * 0.8
        elif self.risk_tolerance == 'Medium':
            adjusted_returns = mean_returns
        else:
            adjusted_returns = mean_returns * 1.2

        logger.info(f"Adjusted expected returns based on risk tolerance '{self.risk_tolerance}': {adjusted_returns}")
        return adjusted_returns

    def portfolio_return(self, weights):
        """
        Calculate the expected portfolio return.

        :param weights: Numpy array of asset weights
        :return: Float representing portfolio return
        """
        ret = np.dot(weights, self.expected_returns)
        logger.debug(f"Portfolio return: {ret}")
        return ret

    def portfolio_volatility(self, weights):
        """
        Calculate the expected portfolio volatility.

        :param weights: Numpy array of asset weights
        :return: Float representing portfolio volatility
        """
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        logger.debug(f"Portfolio volatility: {vol}")
        return vol

    def run_monte_carlo(self, allocations, num_simulations=10000):
        """
        Simulate portfolio returns based on the user's time horizon.

        :param allocations: Numpy array of asset allocations
        :param num_simulations: Integer number of simulations to run
        :return: Numpy array of simulated returns adjusted for time horizon
        """
        returns = self.asset_data.pct_change().dropna().values
        portfolio_returns = np.dot(returns, allocations)
        np.random.seed(42)
        simulation_results = np.random.choice(portfolio_returns, size=num_simulations, replace=True)

        # Adjust simulation results based on time horizon
        simulation_results_adjusted = simulation_results * self.time_horizon_years
        logger.info(f"Monte Carlo simulation completed with {num_simulations} simulations.")
        return simulation_results_adjusted

    def summarize_simulation(self, simulation_results):
        """
        Generate summary statistics from simulation results.

        :param simulation_results: Numpy array of simulation results
        :return: Dictionary containing summary statistics
        """
        summary = {
            'Mean Return': np.mean(simulation_results),
            'Median Return': np.median(simulation_results),
            'Standard Deviation': np.std(simulation_results),
            'Minimum Return': np.min(simulation_results),
            'Maximum Return': np.max(simulation_results),
            '5th Percentile': np.percentile(simulation_results, 5),
            '95th Percentile': np.percentile(simulation_results, 95)
        }
        logger.info(f"Simulation Summary: {summary}")
        return summary