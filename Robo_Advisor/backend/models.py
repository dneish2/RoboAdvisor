# backend/models.py

import pandas as pd
import os
import json
import logging

logger = logging.getLogger(__name__)

class UserProfile:
    def __init__(self, goals, risk_tolerance, available_investment):
        """
        Initialize the UserProfile with goals, risk tolerance, and available investment.

        :param goals: List of dictionaries, each containing 'description', 'amount', and 'time_horizon'
        :param risk_tolerance: String indicating risk level ('Low', 'Medium', 'High')
        :param available_investment: Float indicating the amount available for investment
        """
        self.goals = goals
        self.risk_tolerance = risk_tolerance
        self.available_investment = available_investment

    def save_to_csv(self, filepath):
        try:
            goals_rounded = [
                {**goal, 'time_horizon': round(goal['time_horizon'], 1)} for goal in self.goals
            ]

            df = pd.DataFrame([{
                'goals': json.dumps(goals_rounded),
                'risk_tolerance': self.risk_tolerance,
                'available_investment': self.available_investment
            }])

            write_header = not os.path.exists(filepath)

            df.to_csv(filepath, mode='a', header=write_header, index=False)
            logger.info(f"User profile saved successfully to {filepath}. Headers {'written' if write_header else 'not written'}.")
        except Exception as e:
            logger.error(f"Error saving user profile to CSV: {e}")

    @staticmethod
    def load_from_csv(filepath):
        """
        Load user profiles from a CSV file.

        :param filepath: Path to the CSV file
        :return: Pandas DataFrame containing user profiles
        """
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                expected_columns = ['goals', 'risk_tolerance', 'available_investment']
                if all(col in df.columns for col in expected_columns):
                    logger.info(f"Loaded user profiles from {filepath} with expected columns.")
                    return df
                else:
                    # Handle cases where headers might be missing
                    logger.warning(f"Expected columns missing in {filepath}. Assigning default column names.")
                    df = pd.read_csv(filepath, header=None, names=['goals', 'risk_tolerance', 'available_investment'])
                    return df
            except pd.errors.ParserError as e:
                logger.error(f"ParserError while loading user profiles: {e}")
                # Attempt to read without headers
                logger.info(f"Attempting to read {filepath} without headers.")
                df = pd.read_csv(filepath, header=None, names=['goals', 'risk_tolerance', 'available_investment'])
                return df
            except Exception as e:
                logger.error(f"Error loading user profiles from CSV: {e}")
                return pd.DataFrame()
        else:
            logger.warning(f"File {filepath} does not exist. Returning empty DataFrame.")
            return pd.DataFrame()