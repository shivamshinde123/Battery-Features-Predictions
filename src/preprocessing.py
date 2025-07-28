import re
import os

import pandas as pd
from src.utils import Utility
from sklearn.preprocessing import MinMaxScaler

# setting up the logger
logger = Utility().setup_logger()

class Preprocess:

    """This class is used for preprocessing the data before machine learning model training."""

    def __init__(self):
        pass

    def drop_features(self, df, feature_list):
        """This method drops the features that are not useful for ml prediction model"""
        try:
            df = df.drop(feature_list, axis=1)
            return df
        except Exception as e:
            logger.error('Dropping useless features failed', exc_info=e)
            raise

    def keep_features(self, df, keep_features_list):
        """This method is used to keep the required features and remove the rest"""
        try:
            df = df[keep_features_list]
            return df
        except Exception as e:
            logger.error("Keeping required features only failed", exc_info=e)


    def clean_column(self, name: str):

        """This method renames the feature for easier processing"""

        try:
            # Replace all non-word characters with underscore
            name = re.sub(r"[^\w]", "_", name)

            # Collaspe multiple underscores into one
            name = re.sub(r"_+", "_", name)

            # Strip leading/trailing underscores
            name = name.strip("_")
            return name

        except Exception as e:
            logger.error("Renaming features failed", exc_info=e)
            raise

    def replace_missing_values(self, df):

        """This method is used to replace the missing values in feature with median of the feature"""

        try:
            for feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].median())
            return df
        except Exception as e:
            logger.error("Replacing missing values failed", exc_info=e)
            raise

    def scale_features(self, df):

        """This method is used to scale the features to the similar scale"""

        try:
            scaler = MinMaxScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            return df
        except Exception as e:
            logger.error("Scaling features failed", exc_info=e)
            raise

    def remove_features_with_zero_variance(self, df):

        """This method is used to remove the features with zero variance"""

        try:
            feature_list = list()
            numeric_cols = df.select_dtypes(include='number').columns
            for feature in numeric_cols:
                val = df[feature].var()
                if val == 0:
                    feature_list.append(feature)

            df = df.drop(feature_list, axis=1)
            return df
        except Exception as e:
            logger.error("Removing features with zero variance failed", exc_info=e)
            raise

if __name__ == "__main__":

    # creating object of Preprocess class
    pr = Preprocess()

    # Dataframe
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..'))
    df = pd.read_csv(os.path.join(root_dir, 'Data', 'processed', 'AllTrips.csv'))

    # Dropping useless features
    drop_features= [
    "Time [s]", "max. Battery Temperature [°C]", "displayed SoC [%]", "min. SoC [%]", "max. SoC [%)",
    "Ambient Temperature Sensor [°C]", "Requested Coolant Temperature [°C]", "Temperature Vent right [°C]",
    "Velocity [km/h]]]", "Requested Heating Power [W]"]

    df = pr.drop_features(df, drop_features)
    logger.info("Useless features dropped.")

    # Rewriting the column names properly
    df.columns = [pr.clean_column(col) for col in df.columns]
    logger.info('Column names rewritten properly.')

    # Replacing missing values with median
    df = pr.replace_missing_values(df)
    logger.info("Missing values replaced with median of respective feature.")

    # Dropping duplicate records
    df.drop_duplicates(inplace=True, keep='first')
    logger.info("Removed duplicate records")

    # scaling the features
    df = pr.scale_features(df)
    logger.info("Scaled the features to similar values.")

    # Remove the features with zero variances
    df = pr.remove_features_with_zero_variance(df)
    logger.info("Feature with zero variance removed")

    # Keeping only the prominent features and removing rest
    keep_feature_list = ['Battery_Voltage_V', 'Heating_Power_CAN_kW', 'Heater_Voltage_V', 'Coolant_Volume_Flow_500_l_h', 'Battery_Current_A', 'SoC']

    df = pr.keep_features(df, keep_feature_list)
    logger.info("Most prominent features that give good accuracy without inducing much noise kept and rest dropped.")

    df.to_csv(os.path.join(root_dir, 'Data', 'processed', 'processed_trip_data.csv'), index=False)
    logger.info("Preprocessed data saved as csv file.")