from datetime import timedelta
import pandas as pd
import numpy as np
from src.ERPsalesForecasting.components.data_ingestion import DataIngestion
from src.ERPsalesForecasting.components.data_processing import DataProcessing
from src.ERPsalesForecasting.components.model_training import ModelTraining
from src.ERPsalesForecasting.components.model_selection import ModelSelection
from src.ERPsalesForecasting import logger
from src.ERPsalesForecasting.config.configuration import ConfigurationManager


class SalesForecasting:
    def __init__(self, account_name, account_key, container_name, blob_name, download_file_path):
        self.account_name = account_name
        self.account_key = account_key
        self.container_name = container_name
        self.blob_name = blob_name
        self.download_file_path = download_file_path
        self.models = {}

    def run(self):
        try:
            self.ingest_data()
            self.process_data()
            self.train_and_evaluate_models()
            self.select_best_model()
            self.make_forecasts()
        except Exception as e:
            logger.info(f"An error occurred: {e}")

    def ingest_data(self):
        ingestion = DataIngestion(self.account_name, self.account_key,
                                  self.container_name, self.blob_name, self.download_file_path)
        file_path = ingestion.download_blob_data()
        if file_path is None:
            raise Exception("Failed to download data.")
        self.sales_data = ingestion.load_sales_data()
        if self.sales_data is None:
            raise Exception("Failed to load sales data.")

    def process_data(self):
        processed_data = DataProcessing.preprocess_data(self.sales_data)
        if processed_data is None:
            raise Exception("Failed in data preprocessing.")
        self.aggregated_data = DataProcessing.aggregate_data(processed_data)
        if self.aggregated_data is None:
            raise Exception("Failed in data aggregation.")

    def train_and_evaluate_models(self):
        model_trainer = ModelTraining()
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(
            self.aggregated_data)
        self.training_feature_columns = X_train.columns
        if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
            raise Exception("Data preparation resulted in empty datasets.")

        model_types = ['random_forest', 'gradient_boosting', 'lasso', 'ridge',
                       'elastic_net', 'decision_tree', 'knn', 'svr', 'linear_regression']

        for model_type in model_types:
            logger.info(f"Training and evaluating {model_type} model...")
            model = model_trainer.get_regression_model(model_type)
            if model is None:
                continue
            trained_model = model_trainer.train_model(model, X_train, y_train)
            if trained_model is None:
                continue
            mse, mae, r2 = model_trainer.evaluate_model(
                trained_model, X_test, y_test)
            if None in [mse, mae, r2]:
                continue
            logger.info(f"{model_type} - MSE: {mse}, MAE: {mae}, R2: {r2}")
            self.models[model_type] = (trained_model, mse, mae)

    def select_best_model(self):
        best_model_type, best_model_details = ModelSelection.select_best_model(
            self.models)
        self.best_model = best_model_details['Model']
        logger.info(
            f"The best model is {best_model_type} with MSE: {best_model_details['MSE']} and MAE: {best_model_details['MAE']}")
        return self.best_model

    def forecast_sales_next_week(self, model):
        try:
            last_date = self.aggregated_data['billCreatedDateTime'].max()
            forecast_set = []
            forecast_results = []

            training_feature_columns = self.training_feature_columns

            for product_id, product_name in self.aggregated_data[['ProductID', 'ProductName']].drop_duplicates().values:
                product_sales_data = self.aggregated_data[self.aggregated_data['ProductID'] == product_id]
                product_recent_data = product_sales_data.sort_values(by='billCreatedDateTime', ascending=False)

                # Calculate current inventory based on total sales
                total_sales = product_sales_data['SelledQTY'].sum()
                initial_inventory = product_sales_data['ProductTotalQty'].iloc[0]  # Assuming initial inventory is the first recorded total quantity
                current_inventory = initial_inventory - total_sales

                if not product_recent_data.empty:
                    prev_day_sales = product_recent_data.iloc[0]['SelledQTY']

                    for i in range(1, 8):  # Daily forecast for next week
                        forecast_date = last_date + timedelta(days=i)
                        forecast_df = pd.DataFrame({'ProductID': [product_id], 'PrevDaySales': [prev_day_sales]})
                        forecast_df_encoded = pd.get_dummies(forecast_df, columns=['ProductID'])
                        forecast_df_encoded = forecast_df_encoded.reindex(columns=training_feature_columns, fill_value=0)
                        forecast_sales = model.predict(forecast_df_encoded)
                        forecasted_sales = np.ceil(forecast_sales[0]).astype(int)

                        inventory_after_sales = current_inventory - forecasted_sales
                        reorder_status = 1 if inventory_after_sales <= forecasted_sales else 0

                        forecast_set.append({'ProductID': product_id, 'ProductName': product_name, 
                                             'ForecastDate': forecast_date, 'ForecastedSales': forecasted_sales,
                                             'CurrentInventory': current_inventory, 'InventoryAfterSales': inventory_after_sales,
                                             'ReorderStatus': reorder_status})

                        current_inventory = inventory_after_sales
                        prev_day_sales = forecasted_sales

                    weekly_forecast = sum([f['ForecastedSales'] for f in forecast_set if f['ProductID'] == product_id])
                    forecast_results.append({'ProductID': product_id, 'ProductName': product_name, 
                                             'WeeklyForecastedSales': weekly_forecast, 'FinalInventoryQty': current_inventory,
                                             'ReorderStatus': 1 if current_inventory <= weekly_forecast else 0})

            return pd.DataFrame(forecast_set), pd.DataFrame(forecast_results)
        except Exception as e:
            logger.error(f"Error in forecast_sales_next_week: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def make_forecasts(self):
        try:
            daily_forecasts, weekly_forecasts = self.forecast_sales_next_week(
                self.best_model)
            logger.info("Forecasts generated successfully.")
            print(daily_forecasts)
            print(weekly_forecasts)
        except Exception as e:
            logger.error(f"Error in make_forecasts: {e}")


if __name__ == "__main__":
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    # print(data_ingestion_config)

    account_name = data_ingestion_config.account_name
    account_key = data_ingestion_config.account_key
    container_name = data_ingestion_config.container_name
    download_file_path = data_ingestion_config.download_file_path

    forecasting = SalesForecasting(
        account_name, account_key, container_name, 'data.xlsx', download_file_path)
    forecasting.run()
