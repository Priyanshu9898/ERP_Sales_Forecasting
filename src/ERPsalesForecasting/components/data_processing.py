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
            df = pd.read_csv(dataFilePath)

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
                df = pd.read_csv(self.config.data_file)
                columns = df.columns
                print(columns)

                df['Date'] = pd.to_datetime(df['Date'])
                df.sort_values('Date', inplace=True)

                df.reset_index(inplace=True)

                df.to_csv(self.config.preprocessed_file)

                logger.info(
                    f"Preprocessed file saved at {self.config.preprocessed_file}")

            else:
                logger.info("Data is not available")

        except Exception as e:
            raise e
