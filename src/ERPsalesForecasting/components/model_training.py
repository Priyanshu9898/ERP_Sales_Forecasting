from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from ERPsalesForecasting import logger

class ModelTraining:
    @staticmethod
    def prepare_data(aggregated_data):
        try:
            
            X = pd.get_dummies(aggregated_data[['ProductID', 'PrevDaySales']], columns=['ProductID'])
            y = aggregated_data['SelledQTY']
            
            X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error in preparing data: {e}")
            return None, None, None, None

    def get_regression_model(self, model_type='linear_regression'):
        try:
            if model_type == 'random_forest':
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'gradient_boosting':
                return GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_type == 'lasso':
                return Lasso(alpha=1.0, random_state=42)
            elif model_type == 'ridge':
                return Ridge(alpha=1.0, random_state=42)
            elif model_type == 'elastic_net':
                return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
            elif model_type == 'decision_tree':
                return DecisionTreeRegressor(random_state=42)
            elif model_type == 'knn':
                return KNeighborsRegressor(n_neighbors=5)
            elif model_type == 'svr':
                return SVR()
            elif model_type == 'linear_regression':
                return LinearRegression()
        except Exception as e:
            logger.error(f"Error in getting regression model {model_type}: {e}")
            return None

    @staticmethod
    def train_model(model, X_train, y_train):
        try:
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logger.error(f"Error in training model: {e}")
            return None

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return mse, mae, r2
        except Exception as e:
            logger.error(f"Error in evaluating model: {e}")
            return None, None, None
