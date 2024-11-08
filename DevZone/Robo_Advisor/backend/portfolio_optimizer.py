# backend/portfolio_optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

class PortfolioOptimizer:
    def __init__(self, user_profile):
        # Validate user_profile keys
        required_keys = ['goals', 'risk_tolerance', 'available_investment']
        for key in required_keys:
            if key not in user_profile:
                raise KeyError(f"Missing key in user_profile: '{key}'")

        # Validate 'goals' structure
        if not isinstance(user_profile['goals'], list):
            raise TypeError(f"'goals' should be a list, got {type(user_profile['goals'])}")

        self.user_profile = user_profile
        self.assets = self.define_assets()
        self.asset_data = self.fetch_asset_data()
        self.expected_returns = self.calculate_expected_returns()
        self.cov_matrix = self.asset_data.pct_change().dropna().cov().values

        

    def optimize(self):
        num_assets = len(self.assets)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets,]

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
            portfolio['Investment'] = portfolio['Allocation'] * self.user_profile['available_investment']

            return portfolio, expected_return, simulation_results
        else:
            raise Exception("Optimization failed.")

    def define_assets(self):
        risk = self.user_profile['risk_tolerance']
        if risk == 'High':
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'VOO']
        elif risk == 'Medium':
            return ['VTI', 'VOO', 'AAPL', 'MSFT', 'NVO', 'VRTX']
        else:
            return ['GLD', 'VTI', 'VOO', 'SCHD']
        
        #Power Suggestions
        #Option Contract for high risk
        #Cds for low risk

        #Curate and feed it in a strategy template. For example, only getting into deals with stocks 15-25% off their all time high 
        #in the case of options, in terms of a bundle deal, focus on price efficiency and nutrional value.

    def fetch_asset_data(self):
        # Fetch historical data for assets
        data = {}
        for asset in self.assets:
            if asset == 'BOND':
                # Using a 10-year US Treasury bond as proxy
                ticker = 'TLT'
            elif asset == 'MUTF':
                # Using a bond mutual fund as proxy
                ticker = 'VFIAX'
            else:
                ticker = asset
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y")
            data[asset] = hist['Close']
        df = pd.DataFrame(data)
        df = df.dropna()
        return df

    def calculate_expected_returns(self):
        # Calculate historical mean returns
        mean_returns = self.asset_data.pct_change().mean().values
        # Adjust based on risk tolerance
        risk = self.user_profile['risk_tolerance']
        if risk == 'Low':
            return mean_returns * 0.8
        elif risk == 'Medium':
            return mean_returns
        else:
            return mean_returns * 1.2

    def portfolio_return(self, weights):
        return np.dot(weights, self.expected_returns)

    def portfolio_volatility(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def run_monte_carlo(self, allocations, num_simulations=10000):
        # Simulate portfolio returns
        returns = self.asset_data.pct_change().dropna().values
        portfolio_returns = np.dot(returns, allocations)
        np.random.seed(42)
        simulation_results = np.random.choice(portfolio_returns, size=num_simulations, replace=True)
        return simulation_results
        