from ERPsalesForecasting.config.configuration import ConfigurationManager
from ERPsalesForecasting.components.model_training import ModelTraining
from ERPsalesForecasting import logger


STAGE_NAME = "Model Training and Validation stage"


def train(model_training, X_train_scaled, y_train, X_test_scaled, y_test, dates_test, product_ids_test, model_name="linear_regression"):
    logger.info(f"{model_name} Training started")

    model = model_training.get_regression_model(model_name)
    model.fit(X_train_scaled, y_train)

    model_training.model_evaluation(
        model, model_name, X_test_scaled, y_test, dates_test, product_ids_test)

    logger.info(f"{model_name} Training and evaluation completed successfully")


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        product_encoded_df, numeric_features, df = model_training.data_preprocessing()

        numeric_features['AvailableQtyAfterSell'] = numeric_features['AvailableQtyAfterSell'].apply(
            lambda x: 0 if x < 0 else x)

        X_train, X_test, y_train, y_test, dates_train, dates_test, product_ids_train, product_ids_test = model_training.splitData(
            product_encoded_df, numeric_features, df)
        X_train_scaled, X_test_scaled = model_training.dataScaling(
            X_train, X_test)

        train(model_training, X_train_scaled, y_train, X_test_scaled,
              y_test, dates_test, product_ids_test, "linear_regression")
        train(model_training, X_train_scaled, y_train, X_test_scaled,
              y_test, dates_test, product_ids_test, "decision_tree")
        train(model_training, X_train_scaled, y_train, X_test_scaled,
              y_test, dates_test, product_ids_test, "svr")
        train(model_training, X_train_scaled, y_train, X_test_scaled,
              y_test, dates_test, product_ids_test, "knn")
        train(model_training, X_train_scaled, y_train, X_test_scaled,
              y_test, dates_test, product_ids_test, "gradient_boosting")
        train(model_training, X_train_scaled, y_train, X_test_scaled,
              y_test, dates_test, product_ids_test, "random_forest")
        train(model_training, X_train_scaled, y_train, X_test_scaled,
              y_test, dates_test, product_ids_test, "ridge")
        train(model_training, X_train_scaled, y_train, X_test_scaled,
              y_test, dates_test, product_ids_test, "lasso")
        train(model_training, X_train_scaled, y_train, X_test_scaled,
              y_test, dates_test, product_ids_test, "elastic_net")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e