from azure.storage.blob import BlobServiceClient
import pandas as pd
import os
from ERPsalesForecasting import logger

class DataIngestion:
    def __init__(self, account_name, account_key, container_name, blob_name, download_file_path):
        self.account_name = account_name
        self.account_key = account_key
        self.container_name = container_name
        self.blob_name = blob_name
        self.download_file_path = download_file_path

    def get_connection_string(self):
        return f"DefaultEndpointsProtocol=https;AccountName={self.account_name};AccountKey={self.account_key};EndpointSuffix=core.windows.net"

    def download_blob_data(self):
        try:
            connection_string = self.get_connection_string()
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(container=self.container_name, blob=self.blob_name)
            os.makedirs(os.path.dirname(self.download_file_path), exist_ok=True)
            with open(self.download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            logger.info("Blob data downloaded successfully.")
            return self.download_file_path
        except Exception as e:
            logger.error(f"Error in downloading blob data: {e}")
            return None

    def load_sales_data(self):
        try:
            return pd.read_excel(self.download_file_path)
        except Exception as e:
            logger.error(f"Error in loading sales data: {e}")
            return None
