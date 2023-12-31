import pandas as pd
from ERPsalesForecasting import logger

class DataProcessing:
    @staticmethod
    def preprocess_data(sales_data):
        try:
            sales_data['billCreatedDateTime'] = pd.to_datetime(sales_data['billCreatedDateTime']).dt.date
            sales_data.sort_values(by=['ProductID', 'billCreatedDateTime'], inplace=True)
            sales_data['CumulativeSelledQTY'] = sales_data.groupby('ProductID')['SelledQTY'].cumsum()
            sales_data['CorrectedAvailableQtyAfterSell'] = sales_data['ProductTotalQty'] - sales_data['CumulativeSelledQTY']
            return sales_data
        except Exception as e:
            logger.error(f"Error in preprocessing data: {e}")
            return None

    @staticmethod
    def aggregate_data(sales_data):
        try:
            aggregated_data = sales_data.groupby(['ProductID', 'ProductName', 'billCreatedDateTime']).agg({
                'SelledQTY': 'sum', 
                'ProductTotalQty': 'first', 
                'CorrectedAvailableQtyAfterSell': 'last'
            }).reset_index()
            aggregated_data['PrevDaySales'] = aggregated_data.groupby('ProductID')['SelledQTY'].shift(1).fillna(0)
            return aggregated_data
        except Exception as e:
            logger.error(f"Error in aggregating data: {e}")
            return None
