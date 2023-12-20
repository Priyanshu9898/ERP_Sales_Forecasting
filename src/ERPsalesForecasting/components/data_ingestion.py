from ERPsalesForecasting.entity import DataIngestionConfig
import os
import gdown
from ERPsalesForecasting import logger


class DataIngestion:

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_file(self):
        try:
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_file
            os.makedirs("/artifacts/data_ingestion/", exist_ok=True)
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id, zip_download_dir)
            logger.info(
                f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
