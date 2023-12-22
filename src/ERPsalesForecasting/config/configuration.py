from ERPsalesForecasting.entity import DataIngestionConfig, DataProcessingConfig, ModelTrainingConfig
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
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
        )

        return data_ingestion_config

    def get_data_processing_config(self):
        config = self.config.data_processing

        create_directories([config.root_dir])

        data_processing_config = DataProcessingConfig(
            root_dir=config.root_dir,
            data_file=config.data_file,
            preprocessed_file=config.preprocessed_file,
            isValid=False,
        )

        return data_processing_config

    def get_model_training_config(self):
        config = self.config.model_training
        validationConfig = self.config.model_validation

        create_directories([config.root_dir, validationConfig.root_dir])
        column_names = "ModelName,MSE,MAE,R2\n"
        if (not os.path.exists(validationConfig.result_file) or os.path.getsize(validationConfig.result_file) == 0):
         with open(validationConfig.result_file, "w") as f:
            f.write(column_names)
            logger.info(f"Creating empty file: {validationConfig.result_file}")
            
        else:
            logger.info(f"{validationConfig.result_file} is already exists")

        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            data_file=config.data_file,
            result_file=validationConfig.result_file,
        )

        return model_training_config