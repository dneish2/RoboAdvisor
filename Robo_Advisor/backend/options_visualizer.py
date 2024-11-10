import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class OptionsVisualizer:
    def __init__(self, options_data=None, time_frame=None):
        """
        Initialize OptionsVisualizer with options data and an optional time frame.

        Parameters:
            options_data (DataFrame): DataFrame containing options data
            time_frame (str): Optional time frame for filtering data (e.g., '1 Month')
        """
        self.options_data = options_data
        self.time_frame = time_frame
        if self.options_data is not None:
            self.prepare_data()
            self.filtered_data = self.filter_data_by_time_frame()

    def prepare_data(self):
        logger.info("Preparing options data")
        required_columns = ['Strike', 'Expiration', 'Type', 'Volume', 'Open Interest', 'Last Price']

        missing_columns = [col for col in required_columns if col not in self.options_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.options_data['Expiration'] = pd.to_datetime(self.options_data['Expiration'], errors='coerce')
        self.options_data['Type'] = self.options_data['Type'].str.upper()

        numeric_columns = ['Strike', 'Volume', 'Open Interest', 'Last Price']
        for col in numeric_columns:
            self.options_data[col] = pd.to_numeric(self.options_data[col], errors='coerce')

        self.options_data.dropna(subset=required_columns, inplace=True)

    def filter_data_by_time_frame(self):
        logger.info(f"Filtering data for time frame: {self.time_frame}")
        current_time = pd.Timestamp.now()
        time_frame_mapping = {
            '1 Week': pd.DateOffset(weeks=1),
            '1 Month': pd.DateOffset(months=1),
            '3 Months': pd.DateOffset(months=3),
            '6 Months': pd.DateOffset(months=6)
        }

        end_time = current_time + time_frame_mapping.get(self.time_frame, pd.DateOffset(months=1))
        return self.options_data[self.options_data['Expiration'] <= end_time]

    def plot_options_chain(self):
        """
        Plot the Options Chain Visualization with Strike Price, Volume, and Open Interest.

        Returns:
            Figure: Plotly Figure with the options chain visualized.
        """
        logger.info("Creating options chain visualization")
        if self.filtered_data.empty:
            logger.warning("No data available for options chain visualization")
            return go.Figure()

        # Separate CALL and PUT options
        calls = self.filtered_data[self.filtered_data['Type'] == 'CALL']
        puts = self.filtered_data[self.filtered_data['Type'] == 'PUT']

        # Create subplots for CALL and PUT
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Call Options', 'Put Options'),
            vertical_spacing=0.15
        )

        # Plot CALL options if available
        if not calls.empty:
            fig.add_trace(
                go.Scatter(
                    x=calls['Expiration'],
                    y=calls['Strike'],
                    mode='markers',
                    name='CALLs',
                    marker=dict(
                        size=calls['Volume'].fillna(0)/100 + 5,
                        color=calls['Open Interest'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Open Interest')
                    ),
                    text=[
                        f'Strike: ${strike:,.2f}<br>'
                        f'Volume: {vol:,.0f}<br>'
                        f'Open Interest: {oi:,.0f}<br>'
                        f'Last Price: ${price:,.2f}'
                        for strike, vol, oi, price in zip(
                            calls['Strike'],
                            calls['Volume'],
                            calls['Open Interest'],
                            calls['Last Price']
                        )
                    ],
                    hovertemplate='%{text}<br>Expiration: %{x|%Y-%m-%d}<extra></extra>'
                ),
                row=1, col=1
            )

        # Plot PUT options if available
        if not puts.empty:
            fig.add_trace(
                go.Scatter(
                    x=puts['Expiration'],
                    y=puts['Strike'],
                    mode='markers',
                    name='PUTs',
                    marker=dict(
                        size=puts['Volume'].fillna(0)/100 + 5,
                        color=puts['Open Interest'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Open Interest')
                    ),
                    text=[
                        f'Strike: ${strike:,.2f}<br>'
                        f'Volume: {vol:,.0f}<br>'
                        f'Open Interest: {oi:,.0f}<br>'
                        f'Last Price: ${price:,.2f}'
                        for strike, vol, oi, price in zip(
                            puts['Strike'],
                            puts['Volume'],
                            puts['Open Interest'],
                            puts['Last Price']
                        )
                    ],
                    hovertemplate='%{text}<br>Expiration: %{x|%Y-%m-%d}<extra></extra>'
                ),
                row=2, col=1
            )

        fig.update_layout(
            title='Options Chain Visualization',
            height=800,
            showlegend=True,
            hovermode='closest',
            margin=dict(r=120),
            template='plotly_dark'
        )
        fig.update_xaxes(title_text='Expiration Date', row=1, col=1)
        fig.update_xaxes(title_text='Expiration Date', row=2, col=1)
        fig.update_yaxes(title_text='Strike Price ($)', row=1, col=1)
        fig.update_yaxes(title_text='Strike Price ($)', row=2, col=1)

        logger.info("Options chain visualization created successfully")
        return fig

    def plot_option_history(self, option_history, option_type):
        """
        Plot historical price for a specific option contract.

        Parameters:
            option_history (DataFrame): Contains 'Close' column.
            option_type (str): 'CALL' or 'PUT'

        Returns:
            Figure: Plotly Figure with the historical data visualized.
        """
        if option_history.empty:
            logger.warning(f"No historical data provided for {option_type} option.")
            return go.Figure()

        fig = go.Figure()

        # Plot Price History
        fig.add_trace(
            go.Scatter(
                x=option_history.index,
                y=option_history['Close'],
                mode='lines',
                name=f'{option_type} Close Price',
                line=dict(color='green' if option_type.upper() == 'CALL' else 'red', width=2)
            )
        )

        # Enhance the layout for better aesthetics
        fig.update_layout(
            title=f"{option_type} Option Close Price History",
            xaxis_title="Date",
            yaxis_title="Close Price ($)",
            template='plotly_dark',
            hovermode='x unified'
        )

        return fig