from flask import Flask, jsonify
from flask_restx import Api, Resource
from flask_cors import CORS, cross_origin
from main import SalesForecasting
import traceback
from src.ERPsalesForecasting.config.configuration import ConfigurationManager

app = Flask(__name__)
api = Api(app, version='1.0', title='Sales Forecasting API', description='API for Sales Forecasting')

# List of allowed origins
allowed_origins = ["http://localhost:3000/"]

# Enable CORS for the entire app with specific settings
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"], allow_headers="*")


config = ConfigurationManager()
data_ingestion_config = config.get_data_ingestion_config()

    
account_name=data_ingestion_config.account_name
account_key=data_ingestion_config.account_key
container_name=data_ingestion_config.container_name
download_file_path=data_ingestion_config.download_file_path
    
forecasting = SalesForecasting(account_name, account_key, container_name, 'data.xlsx', download_file_path)

ns = api.namespace('forecasting', description='Operations related to Sales Forecasting')

@ns.route('/forecast')
class TrainModel(Resource):
    @cross_origin(origins=allowed_origins)
    @ns.doc(responses={200: 'Success', 500: 'Internal Server Error'})
    def get(self):
        '''Train the model and get initial forecasts'''
        try:
            forecasting.run()
            daily_forecast, weekly_forecast = forecasting.forecast_sales_next_week(forecasting.best_model)
            return jsonify({
                "daily_forecast": daily_forecast.to_dict(orient='records'),
                "weekly_forecast": weekly_forecast.to_dict(orient='records')
            }), 200
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

# @ns.route('/forecast/nextday')
# class NextDayForecast(Resource):
#     @cross_origin(origins=allowed_origins)
#     @ns.doc(responses={200: 'Success', 500: 'Internal Server Error'})
#     def get(self):
#         '''Get the forecast for the next day'''
#         try:
#             daily_forecast, _ = forecasting.forecast_sales_next_week(forecasting.best_model)
#             return jsonify(daily_forecast.head().to_dict(orient='records')), 200
#         except Exception as e:
#             traceback.print_exc()
#             return jsonify({"error": str(e)}), 500

# @ns.route('/forecast/weekly')
# class WeeklyForecast(Resource):
#     @cross_origin(origins=allowed_origins)
#     @ns.doc(responses={200: 'Success', 500: 'Internal Server Error'})
#     def get(self):
#         '''Get the weekly forecast'''
#         try:
#             _, weekly_forecast = forecasting.forecast_sales_next_week(forecasting.best_model)
#             return jsonify(weekly_forecast.head().to_dict(orient='records')), 200
#         except Exception as e:
#             traceback.print_exc()
#             return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)