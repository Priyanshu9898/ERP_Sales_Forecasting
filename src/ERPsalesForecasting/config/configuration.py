from ERPsalesForecasting.entity import DataIngestionConfig
from ERPsalesForecasting.constants import *
from ERPsalesForecasting.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(self, config_file_path=CONFIG_FILE_PATH, param_file_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.param = read_yaml(param_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
        )

        return data_ingestion_config
