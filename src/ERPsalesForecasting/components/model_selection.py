import pandas as pd
import logging
from ERPsalesForecasting import logger


class ModelSelection:
    @staticmethod
    def select_best_model(models):
        try:
            # Create a DataFrame from the models' evaluation metrics
            df = pd.DataFrame.from_dict(models, orient='index', columns=[
                                        'Model', 'MSE', 'MAE'])

            # Sort the DataFrame first by MSE, then by MAE
            sorted_models = df.sort_values(by=['MSE', 'MAE'])

            # Get the best model's details
            best_model_type = sorted_models.index[0]
            best_model_details = sorted_models.iloc[0]

            return best_model_type, best_model_details
        
        except Exception as e:
            logger.error(f"Error in selecting the best model: {e}")
            return None, None
