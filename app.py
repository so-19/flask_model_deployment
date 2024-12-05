import os
import pickle
import logging
import traceback
from flask import Flask, request, jsonify
import numpy as np
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Fahrenheit to Celsius conversion
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

# Scaler class to normalize the input data
class WeatherScaler:
    def __init__(self):
        self.feature_ranges = {
            'temperature': (19, 110),  # Fahrenheit
            'humidity': (20, 90),
            'wind_speed': (0, 50),
            'precipitation': (0, 20)
        }
    
    def transform(self, data):
        scaled_data = []
        for value, (min_val, max_val) in zip(data, self.feature_ranges.values()):
            scaled_value = (value - min_val) / (max_val - min_val)
            scaled_value = max(0, min(1, scaled_value))
            scaled_data.append(scaled_value)
        return np.array(scaled_data)
    
    def inverse_transform(self, scaled_data):
        unscaled_data = []
        for scaled_value, (min_val, max_val) in zip(scaled_data, self.feature_ranges.values()):
            unscaled_value = scaled_value * (max_val - min_val) + min_val
            if min_val == 0 and max_val == 20:
                unscaled_value = max(0, unscaled_value)
            unscaled_data.append(unscaled_value)
        return np.array(unscaled_data)

# WeatherPipeline class to handle prediction logic
class WeatherPipeline:
    def __init__(self, sequence_length=1):
        self.sequence_length = sequence_length
        self.num_features = 4  # temperature, humidity, wind speed, precipitation
        self.output_days = 5

        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'final_2_advanced_weather_prediction_model.h5')
            logger.info(f"Attempting to load model from: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at path: {model_path}")
                raise FileNotFoundError(f"Model file not found at path: {model_path}")
            
            # Load model using pickle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info("Model loaded successfully!")
        except FileNotFoundError as fnf_error:
            logger.error(f"FileNotFoundError: {str(fnf_error)}")
            raise Exception(f"File not found: {str(fnf_error)}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error("Stack trace: " + traceback.format_exc())
            raise Exception(f"Failed to load model: {str(e)}")
        
        self.scaler = WeatherScaler()
    
    def prepare_input(self, input_data):
        if len(input_data) != self.num_features:
            raise ValueError(f"Input data must contain {self.num_features} features")
        
        scaled_data = self.scaler.transform(input_data)
        # Reshape to (1, sequence_length, num_features)
        model_input = scaled_data.reshape(1, self.sequence_length, self.num_features)
        return model_input
    
    def predict(self, input_data):
        model_input = self.prepare_input(input_data)
        scaled_predictions = self.model.predict(model_input)
        
        # Reshape predictions to (output_days, num_features)
        scaled_predictions = scaled_predictions.reshape(self.output_days, self.num_features)
        
        # Inverse transform each day's predictions
        original_predictions = np.array([
            self.scaler.inverse_transform(day_pred)
            for day_pred in scaled_predictions
        ])
        
        predictions = []
        feature_names = ['temperature', 'humidity', 'wind_speed', 'precipitation']
        start_date = datetime.now() + timedelta(days=1)
        
        for day in range(self.output_days):
            current_date = start_date + timedelta(days=day)
            original_values = {}
            for i, name in enumerate(feature_names):
                value = float(original_predictions[day][i])
                if name == 'temperature':
                    value = fahrenheit_to_celsius(value)
                original_values[name] = round(value, 2)
            
            day_pred = {
                'date': current_date.strftime('%Y-%m-%d'),
                'day': current_date.strftime('%A'),
                'scaled_values': {
                    name: float(scaled_predictions[day][i])
                    for i, name in enumerate(feature_names)
                },
                'original_values': original_values
            }
            predictions.append(day_pred)
        
        return predictions

# Flask app
app = Flask(__name__)

pipeline = None
try:
    logger.info("Initializing WeatherPipeline...")
    pipeline = WeatherPipeline()
except Exception as e:
    logger.error(f"Error initializing pipeline: {str(e)}")
    logger.error(traceback.format_exc())

@app.route('/')
def home():
    return "Welcome to the Weather Prediction API! Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        error_msg = 'Model failed to load. Please check server logs.'
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500
    
    try:
        input_data = request.json.get('inputs')
        if not input_data:
            return jsonify({'error': 'No input data provided. Expected JSON: {"inputs": [...]}'}), 400

        logger.info(f"Received input data: {input_data}")

        if len(input_data) != 4:
            logger.error("Incorrect number of features in input data.")
            return jsonify({'error': 'Input data must contain 4 features (temperature, humidity, wind_speed, precipitation).'}), 400

        predictions = pipeline.predict(input_data)
        response = {'predictions': predictions}
        logger.info(f"Predictions: {predictions}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error("Stack trace: " + traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
