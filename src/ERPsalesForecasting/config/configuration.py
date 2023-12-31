from ERPsalesForecasting.entity import DataIngestionConfig
from ERPsalesForecasting.constants import *
from ERPsalesForecasting.utils.common import read_yaml, create_directories
import os
from ERPsalesForecasting import logger


class ConfigurationManager:
    def __init__(self, config_file_path=CONFIG_FILE_PATH, param_file_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.param = read_yaml(param_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            account_name=config.account_name,
            account_key=config.account_key,
            container_name=config.container_name,
            download_file_path=Path(config.download_file_path),
        )

        return data_ingestion_config
