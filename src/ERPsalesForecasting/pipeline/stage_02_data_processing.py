from ERPsalesForecasting.config.configuration import ConfigurationManager
from ERPsalesForecasting.components.data_processing import DataProcessing
from ERPsalesForecasting import logger


STAGE_NAME = "Data Validation & Processing stage"


class DataProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_processing_config = config.get_data_processing_config()
        data_processing = DataProcessing(config=data_processing_config)
        data_processing.data_validation()
        data_processing.prepare_data()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataProcessingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
