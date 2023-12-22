from ERPsalesForecasting.entity import DataProcessingConfig
import os
import gdown
from ERPsalesForecasting import logger
from pathlib import Path
import pandas as pd
from ERPsalesForecasting.utils.common import create_directories, get_size


class DataProcessing:

    def __init__(self, config: DataProcessingConfig) -> None:
        self.config = config

    def data_validation(self) -> None:
        dataFilePath = Path(self.config.data_file)

        try:
            df = pd.read_excel(dataFilePath)

            if (get_size(dataFilePath) != ""):
                logger.info(f"Dataset is available at: {dataFilePath}")
                logger.info(f"Dataset size: {df.shape}")
                logger.info(f"Columns in Dataset {df.columns}")
                self.config.isValid = True
        except Exception as e:
            raise e

    def prepare_data(self) -> None:

        create_directories([self.config.root_dir])

        try:
            if (self.config.isValid):
                df = pd.read_excel(self.config.data_file)
                columns = df.columns
                print(columns)

                df['Date'] = pd.to_datetime(
                    df['billCreatedDateTime'], dayfirst=True)
                df.sort_values('Date', inplace=True)

                df.reset_index(inplace=True)

                product_id_column = 'ProductID'
                date_column = 'Date'
                quantity_sold_column = 'SelledQTY'
                product_name_column = 'ProductName'
                total_qty_column = 'ProductTotalQty'

                df[date_column] = pd.to_datetime(df[date_column]).dt.date

                # Calculate the cumulative quantity sold for each product
                df['CumulativeSelledQTY'] = df.groupby([product_id_column])[
                    quantity_sold_column].cumsum()

                # Group by ProductID and Date, and calculate required fields
                merged_data = df.groupby([product_id_column, date_column]).agg({
                    product_name_column: 'first',
                    total_qty_column: 'max',
                    quantity_sold_column: 'sum',
                    'CumulativeSelledQTY': 'max'
                }).reset_index()

                # Calculate AvailableQtyAfterSell
                merged_data['AvailableQtyAfterSell'] = merged_data[total_qty_column] - \
                    merged_data['CumulativeSelledQTY']

                # Drop the CumulativeSelledQTY column
                merged_data.drop(columns=['CumulativeSelledQTY'], inplace=True)

                # Save file
                merged_data.to_csv(self.config.preprocessed_file, index=False)

                logger.info(
                    f"Preprocessed sales file saved at {self.config.preprocessed_file}")

            else:
                logger.info("Data is not available")

        except Exception as e:
            raise e
