from flask import Flask, render_template, request, jsonify, session
import numpy as np
from energy_predictor import EnergyPredictor
import logging
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production!

# Initialize the predictor
predictor = EnergyPredictor()

# Try to load models on startup
if not predictor.load_models():
    logging.info("No pre-trained models found. Training new models...")
    for etype in ['solar', 'wind', 'hydro']:
        predictor.train_energy_model(etype, num_samples=100)
    predictor.save_models()
    logging.info("Training completed and models saved!")
else:
    logging.info("Successfully loaded pre-trained models")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Get input parameters
        location_type = data.get('location_type', 'coordinates')
        energy_type = data.get('energy_type', 'solar')
        area = float(data.get('area', 100))
        
        if location_type == 'coordinates':
            lat = float(data['latitude'])
            lon = float(data['longitude'])
            result = predictor.predict_energy(lat, lon, energy_type, area)
            location_info = f"Coordinates ({lat:.4f}, {lon:.4f})"
            
        else:  # city name
            city_name = data['city_name']
            result = predictor.predict_energy_city(city_name, energy_type, area)
            location_info = city_name
        
        # Format the response
        response = {
            'success': True,
            'location': location_info,
            'energy_type': energy_type,
            'area': area,
            'prediction': {
                'energy_kwh': round(result['predicted_energy_kwh'], 2),
                'suitability_score': round(result['suitability_score'], 3),
                'confidence': round(result['confidence'], 3),
                'annual_energy_mwh': round(result['predicted_energy_kwh'] / 1000, 2),
                'daily_energy_kwh': round(result['predicted_energy_kwh'] / 365, 2)
            },
            'metrics': result['input_data']['metrics']
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/compare', methods=['POST'])
def compare():
    try:
        data = request.get_json()
        location_type = data.get('location_type', 'coordinates')
        area = float(data.get('area', 100))
        
        results = {}
        
        if location_type == 'coordinates':
            lat = float(data['latitude'])
            lon = float(data['longitude'])
            location_info = f"({lat:.4f}, {lon:.4f})"
            
            for energy_type in ['solar', 'wind', 'hydro']:
                result = predictor.predict_energy(lat, lon, energy_type, area)
                results[energy_type] = {
                    'energy_kwh': round(result['predicted_energy_kwh'], 2),
                    'suitability': round(result['suitability_score'], 3),
                    'confidence': round(result['confidence'], 3)
                }
                
        else:  # city name
            city_name = data['city_name']
            location_info = city_name
            
            for energy_type in ['solar', 'wind', 'hydro']:
                result = predictor.predict_energy_city(city_name, energy_type, area)
                results[energy_type] = {
                    'energy_kwh': round(result['predicted_energy_kwh'], 2),
                    'suitability': round(result['suitability_score'], 3),
                    'confidence': round(result['confidence'], 3)
                }
        
        return jsonify({
            'success': True,
            'location': location_info,
            'area': area,
            'comparison': results
        })
        
    except Exception as e:
        logging.error(f"Comparison error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)