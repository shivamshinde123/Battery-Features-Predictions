from sys import exc_info

import mlflow
import xgboost as xgb
from utils import Utility
import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

logger = Utility().setup_logger()

class Models:

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(os.path.join(self.script_dir, '..'))
        self.df = pd.read_csv(os.path.join(self.root_dir, 'Data', 'processed', 'processed_trip_data.csv'))

    def adjusted_r2_score(self, y_true, y_pred, n_features):
        """
        Compute Adjusted R² score.

        Parameters:
        - y_true: Ground truth values
        - y_pred: Predicted values
        - n_features: Number of predictors/features used

        Returns:
        - Adjusted R² score
        """
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)  # number of observations
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adj_r2

    def train_xgboost(self):

        """This method is used to train the xgboost model"""

        try:

            u = Utility()

            params = u.read_params()

            with mlflow.start_run():

                n_estimators = params['Models']['xgb']['n_estimators']
                max_depth = params['Models']['xgb']['max_depth']
                learning_rate = params['Models']['xgb']['learning_rate']
                subsample = params['Models']['xgb']['subsample']
                colsample_bytree = params['Models']['xgb']['colsample_bytree']
                random_state = params['General']['random_state']

                mlflow.log_params({
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree
                })

                X, y = self.df.drop('SoC', axis=1), self.df['SoC']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=23)

                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=random_state
                )

                model.fit(X_train, y_train)

                logger.info("xgboost model training completed.")

                y_pred_test = model.predict(X_test)
                y_pred_val = model.predict(X_val)

                rmse_test = root_mean_squared_error(y_test, y_pred_test)
                r2_adjusted_score_test = self.adjusted_r2_score(y_test, y_pred_test, n_features=len(X_train.columns))

                rmse_val = root_mean_squared_error(y_val, y_pred_val)
                r2_adjusted_score_val = self.adjusted_r2_score(y_val, y_pred_val, n_features=len(X_train.columns))

                logger.info("trained xgboost model tested on the test data.")

                mlflow.log_metric('xgb_rmse_test', rmse_test)
                mlflow.log_metric('xgb_r2_adjusted_score_test', r2_adjusted_score_test)

                mlflow.log_metric('xgb_rmse_val', rmse_val)
                mlflow.log_metric('xgb_r2_adjusted_score_val', r2_adjusted_score_val)

                os.makedirs(os.path.join(self.root_dir, 'Metrics'), exist_ok=True)

                xgb_metrics = {
                    'rmse_test' : rmse_test,
                    'rmse_val': rmse_val,
                    'r2_adjusted_score_test': r2_adjusted_score_test,
                    'r2_adjusted_score_val': r2_adjusted_score_val
                }

                with open(os.path.join(self.root_dir, 'Metrics', 'xgb_metrics.json'), 'w') as json_file:
                    json.dump(xgb_metrics, json_file, indent=4)

                os.makedirs(os.path.join(self.root_dir, 'Models'), exist_ok=True)

                with open(os.path.join(self.root_dir, "Models", "xgb_model.pkl"), "wb") as f:
                    pickle.dump(model, f)

                logger.info("trained xgboost model saved.")

        except Exception as e:
            logger.error("xgboost model training failed", exc_info=e)
            raise

    def train_nn(self):
        pass


if __name__ == "__main__":

    models = Models()
    models.train_xgboost()


