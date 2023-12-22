import pandas as pd
import numpy as np
from ERPsalesForecasting import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from ERPsalesForecasting.entity import ModelTrainingConfig


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig) -> None:
        self.config = config

    def data_preprocessing(self):

        df = pd.read_csv(self.config.data_file)

        encoder = OneHotEncoder(sparse=False)
        product_encoded = encoder.fit_transform(df[['ProductID']])
        product_encoded_df = pd.DataFrame(product_encoded, columns=[
                                          f'Product_{i}' for i in range(product_encoded.shape[1])])

        numeric_features = df.select_dtypes(
            include=[np.number]).drop(['SelledQTY'], axis=1)
        numeric_features.drop(['ProductTotalQty'], axis=1, inplace=True)
        return (product_encoded_df, numeric_features, df)

    def splitData(self, numeric_features, product_encoded_df, df):
        X = pd.concat([numeric_features, product_encoded_df], axis=1)
        y = df['SelledQTY']

        dates = df['Date']
        product_ids = df['ProductID']

        X_train, X_test, y_train, y_test, dates_train, dates_test, product_ids_train, product_ids_test = train_test_split(
            X, y, dates, product_ids, test_size=0.2, random_state=42)

        return (X_train, X_test, y_train, y_test, dates_train, dates_test, product_ids_train, product_ids_test)

    def dataScaling(self, X_train, X_test):
        # Normalize the Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def get_regression_model(self, model_type='linear'):
        if model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        # elif model_type == 'xgboost':
        #     return xgb(n_estimators=100, random_state=42)
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
            from sklearn.linear_model import LinearRegression
            return LinearRegression()

    def model_evaluation(self, model, model_name, X_test_scaled, y_test, dates_test, product_ids_test):
        y_pred = model.predict(X_test_scaled)

        # print(y_pred)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Model: {model_name}, MSE: {mse}, MAE: {mae}, R2 Score: {r2}')

        logger.info(
            'Model: {model_name}, MSE: {mse}, MAE: {mae}, R2 Score: {r2}')

        # Your code for updating the DataFrame
        new_data = pd.DataFrame({
            'ModelName': [model_name],
            'MSE': [mse],
            'MAE': [mae],
            'R2': [r2]
        })

        new_row = [model_name, mse, mae, r2]

        result_df = pd.read_csv(self.config.result_file)

        if result_df['ModelName'].isin([model_name]).any():
            result_df.loc[result_df['ModelName'] == model_name,
                          ['MSE', 'MAE', 'R2']] = [mse, mae, r2]

        else:
            result_df = result_df.append(new_data, ignore_index=True)

        # with open(self.config.result_file, 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     # Write the new data
        #     writer.writerow(new_row)
            # result_df = pd.concat([result_df, new_data], ignore_index=True)

        logger.info(
            'Model: {model_name}, evaluation data has been updated to results file')

        comparison_df = pd.DataFrame({
            'Date': dates_test,
            'ProductId': product_ids_test,
            'ActualSales': y_test,
            'PredictedSales': y_pred
        })

        comparison_df.sort_values(by=['Date', 'ProductId'], inplace=True)

        logger.info(f'{comparison_df.head()}')

        comparison_df.to_csv(f'{self.config.root_dir}/{model_name}.csv')

        logger.info(
            f'test result of {model_name} is saved to: {self.config.root_dir}/{model_name}.csv')

    def best_model(self):

        df = pd.read_csv(self.config.result_file)
        sorted_data = df.sort_values(by=['MSE', 'MAE'])

        # Get the top model (lowest MSE and MAE)
        best_model = sorted_data.iloc[0]

        print("Best model based on lowest MSE and MAE:")
        print(best_model['ModelName'])

        logger.info(f"{best_model['ModelName']} is the best model")

        return best_model['ModelName']
