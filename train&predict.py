import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import requests
from typing import Dict, Any, List, Tuple
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

np.random.seed(42)  # Reproducible random numbers

class EnergyPredictor:
    def __init__(self):
        self.models: Dict[str, RandomForestRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.features: Dict[str, List[str]] = {
            'solar': ['latitude', 'longitude', 'elevation', 'annual_irradiance', 
                     'avg_temperature', 'avg_humidity', 'cloud_cover'],
            'wind': ['latitude', 'longitude', 'elevation', 'avg_wind_speed',
                    'wind_direction', 'temperature', 'air_density'],
            'hydro': ['latitude', 'longitude', 'elevation', 'precipitation',
                     'temperature', 'slope', 'drainage_area']
        }

    # ------------------ 📡 DATA FETCHING ------------------
    def fetch_combined_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetch and combine environmental data for a given location."""
        try:
            nasa_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            nasa_params = {
                'parameters': 'ALLSKY_SFC_SW_DWN,T2M,RH2M,CLOUD_AMT,CLRSKY_DAYS',
                'start': '20220101',
                'end': '20221231',
                'latitude': lat,
                'longitude': lon,
                'community': 'RE',
                'format': 'JSON'
            }

            response = requests.get(nasa_url, params=nasa_params, timeout=15)
            response.raise_for_status()
            nasa_data = response.json()

            processed = self.process_nasa_format(nasa_data, lat, lon)
            return processed

        except Exception as e:
            logging.warning(f"Data fetch failed for ({lat:.2f}, {lon:.2f}) — using fallback. Error: {e}")
            return self.generate_fallback_data(lat, lon)

    def process_nasa_format(self, nasa_data: Dict[str, Any], lat: float, lon: float) -> Dict[str, Any]:
        """Process NASA API response into usable metrics."""
        if 'properties' not in nasa_data:
            return self.generate_fallback_data(lat, lon)

        parameters = nasa_data['properties'].get('parameter', {})

        # Extract arrays
        irr_vals = [v for v in parameters.get('ALLSKY_SFC_SW_DWN', {}).values() if v is not None]
        temp_vals = [v for v in parameters.get('T2M', {}).values() if v is not None]
        hum_vals = [v for v in parameters.get('RH2M', {}).values() if v is not None]
        
        # Improved cloud cover calculation
        cloud_cover_vals = [v for v in parameters.get('CLOUD_AMT', {}).values() if v is not None]
        if cloud_cover_vals:
            cloud_cover = np.mean(cloud_cover_vals) / 100  # Convert percentage to ratio
        else:
            # Fallback calculation based on clear sky days
            clr_vals = [v for v in parameters.get('CLRSKY_DAYS', {}).values() if v is not None]
            avg_clear_days = np.mean(clr_vals) if clr_vals else 200
            cloud_cover = max(0.0, min(1.0, 1 - avg_clear_days / 365))

        return {
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat, self.get_elevation(lat, lon)]
            },
            "metrics": {
                "annual_irradiance": np.mean(irr_vals) * 365 if irr_vals else 1600,
                "avg_temperature": np.mean(temp_vals) if temp_vals else 15,
                "avg_humidity": np.mean(hum_vals) if hum_vals else 65,
                "cloud_cover": cloud_cover
            },
            "wind_speed": 5 + (abs(lat) - 30) * 0.1,
            "precipitation": 1000 - abs(lat) * 10,
            "data_quality": "api"  # Track data source quality
        }

    def generate_fallback_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Generate default values when data API fails."""
        return {
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat, self.get_elevation(lat, lon)]
            },
            "metrics": {
                "annual_irradiance": max(800, 1600 - abs(lat) * 20),
                "avg_temperature": 20 - abs(lat) * 0.5,
                "avg_humidity": 60 + abs(lat) * 0.3,
                "cloud_cover": 0.4
            },
            "wind_speed": 5 + (abs(lat) - 30) * 0.1,
            "precipitation": 1000 - abs(lat) * 10,
            "data_quality": "fallback"  # Indicate this is simulated data
        }

    def get_elevation(self, lat: float, lon: float) -> float:
        """Mock elevation function (replace with real API for accuracy)."""
        # Example: vary elevation smoothly with latitude
        return 100 + np.sin(np.radians(lat)) * 500

    # ------------------ 🧠 TRAINING ------------------
    def train_energy_model(self, energy_type: str, num_samples: int = 1000):
        """Train model for specific energy type using simulated global samples."""
        logging.info(f"Training {energy_type.capitalize()} model on {num_samples} samples...")

        features_list = []
        targets = []

        for i in range(num_samples):
            # Progress tracking
            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i + 1}/{num_samples} samples for {energy_type}")
                
            lat = np.random.uniform(-60, 60)
            lon = np.random.uniform(-180, 180)
            data = self.fetch_combined_data(lat, lon)
            elev = data['geometry']['coordinates'][2]

            if energy_type == 'solar':
                feats = [lat, lon, elev,
                         data['metrics']['annual_irradiance'],
                         data['metrics']['avg_temperature'],
                         data['metrics']['avg_humidity'],
                         data['metrics']['cloud_cover']]
                target = data['metrics']['annual_irradiance'] * 0.15 * 0.75 * 100

            elif energy_type == 'wind':
                feats = [lat, lon, elev, data['wind_speed'], 270,
                         data['metrics']['avg_temperature'], 1.225]
                target = 0.5 * 1.225 * 100 * (data['wind_speed'] ** 3) * 0.35 * 8760 * 0.3

            elif energy_type == 'hydro':
                feats = [lat, lon, elev, data['precipitation'],
                         data['metrics']['avg_temperature'], 10, 1000]
                target = data['precipitation'] * 1000 * 9.81 * 10 * 0.8 / 31536000 * 8760

            features_list.append(feats)
            targets.append(max(0, target))

        X = np.array(features_list)
        y = np.array(targets)

        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_scaled, y)

        # Enhanced evaluation
        preds = model.predict(X_scaled)
        r2 = r2_score(y, preds)
        rmse = mean_squared_error(y, preds, squared=False)
        
        # Calculate feature importance
        feature_importance = model.feature_importances_
        top_feature_idx = np.argmax(feature_importance)
        top_feature = self.features[energy_type][top_feature_idx]
        
        logging.info(f"{energy_type.capitalize()} Model — R²: {r2:.3f}, RMSE: {rmse:.2f}")
        logging.info(f"Top feature: {top_feature} ({feature_importance[top_feature_idx]:.3f})")

        self.scalers[energy_type] = scaler
        self.models[energy_type] = model

    # ------------------ 📈 PREDICTION ------------------
    def predict_energy(self, lat: float, lon: float, energy_type: str, area: float = 100) -> Dict[str, Any]:
        """Predict energy potential for a location and energy type."""
        if energy_type not in self.models:
            raise ValueError(f"No trained model found for '{energy_type}'. Train or load one first.")

        data = self.fetch_combined_data(lat, lon)
        elev = data['geometry']['coordinates'][2]

        if energy_type == 'solar':
            feats = [[lat, lon, elev,
                      data['metrics']['annual_irradiance'],
                      data['metrics']['avg_temperature'],
                      data['metrics']['avg_humidity'],
                      data['metrics']['cloud_cover']]]
        elif energy_type == 'wind':
            feats = [[lat, lon, elev, data['wind_speed'], 270,
                      data['metrics']['avg_temperature'], 1.225]]
        elif energy_type == 'hydro':
            feats = [[lat, lon, elev, data['precipitation'],
                      data['metrics']['avg_temperature'], 10, 1000]]

        feats_scaled = self.scalers[energy_type].transform(feats)
        base_pred = self.models[energy_type].predict(feats_scaled)[0]

        # Calculate confidence based on data quality
        confidence = self.calculate_prediction_confidence(data, energy_type)
        
        adjusted = base_pred * (area / 100)
        suitability = min(1.0, adjusted / (area * 200))

        return {
            'predicted_energy_kwh': round(adjusted, 2),
            'suitability_score': round(suitability, 3),
            'confidence': round(confidence, 3),
            'input_data': data,
            'energy_type': energy_type,
            'area_used_m2': area
        }

    def calculate_prediction_confidence(self, data: Dict[str, Any], energy_type: str) -> float:
        """Calculate prediction confidence based on data quality."""
        base_confidence = 0.85
        
        # Reduce confidence for fallback data
        if data.get('data_quality') == 'fallback':
            base_confidence *= 0.7
        
        metrics = data.get('metrics', {})
        
        # Energy type specific confidence adjustments
        if energy_type == 'solar':
            if metrics.get('annual_irradiance', 0) < 500:  # Very low irradiance
                base_confidence *= 0.8
            if metrics.get('cloud_cover', 0) > 0.8:  # High cloud cover
                base_confidence *= 0.9
                
        elif energy_type == 'wind':
            wind_speed = data.get('wind_speed', 0)
            if wind_speed < 3:  # Very low wind
                base_confidence *= 0.8
            elif wind_speed > 12:  # Extreme wind
                base_confidence *= 0.9
                
        elif energy_type == 'hydro':
            precipitation = data.get('precipitation', 0)
            if precipitation < 500:  # Very dry area
                base_confidence *= 0.7
        
        return min(0.95, max(0.6, base_confidence))  # Keep within reasonable bounds

    # ------------------ 💾 SAVE / LOAD ------------------
    def save_models(self, base_path='energy_models'):
        """Save all trained models with metadata."""
        os.makedirs(base_path, exist_ok=True)
        for etype in self.models:
            model_data = {
                'model': self.models[etype],
                'scaler': self.scalers[etype],
                'features': self.features[etype],
                'training_date': np.datetime64('now'),
                'energy_type': etype
            }
            joblib.dump(model_data, f'{base_path}/{etype}_model.joblib')
            logging.info(f"Saved {etype} model to '{base_path}/{etype}_model.joblib'")
        
        # Save overall metadata
        metadata = {
            'energy_types': list(self.models.keys()),
            'total_models': len(self.models),
            'save_timestamp': np.datetime64('now')
        }
        joblib.dump(metadata, f'{base_path}/metadata.joblib')
        logging.info(f"All models saved to '{base_path}/'")

    def load_models(self, base_path='energy_models'):
        """Load trained models with error handling."""
        try:
            metadata = joblib.load(f'{base_path}/metadata.joblib')
            logging.info(f"Loading {metadata['total_models']} models from {base_path}")
        except FileNotFoundError:
            logging.warning("No metadata found, loading individual models...")
        
        loaded_count = 0
        for etype in ['solar', 'wind', 'hydro']:
            try:
                model_data = joblib.load(f'{base_path}/{etype}_model.joblib')
                self.models[etype] = model_data['model']
                self.scalers[etype] = model_data['scaler']
                self.features[etype] = model_data['features']
                training_date = model_data.get('training_date', 'unknown')
                logging.info(f"✓ Loaded {etype} model (trained: {training_date})")
                loaded_count += 1
            except FileNotFoundError:
                logging.warning(f"✗ No saved model found for {etype}")
        
        logging.info(f"Successfully loaded {loaded_count}/3 models")

    # ------------------ 🔍 ANALYSIS ------------------
    def analyze_location(self, lat: float, lon: float, area: float = 100) -> Dict[str, Any]:
        """Comprehensive analysis of all energy types for a location."""
        results = {}
        
        for energy_type in ['solar', 'wind', 'hydro']:
            if energy_type in self.models:
                try:
                    prediction = self.predict_energy(lat, lon, energy_type, area)
                    results[energy_type] = prediction
                except Exception as e:
                    logging.error(f"Error predicting {energy_type} energy: {e}")
                    results[energy_type] = {'error': str(e)}
        
        # Rank by suitability
        ranked_results = sorted(
            [(k, v) for k, v in results.items() if 'error' not in v],
            key=lambda x: x[1]['suitability_score'],
            reverse=True
        )
        
        return {
            'location': {'latitude': lat, 'longitude': lon, 'area_m2': area},
            'predictions': results,
            'recommendations': ranked_results,
            'best_option': ranked_results[0] if ranked_results else None
        }

    def batch_predict(self, locations: List[Tuple[float, float, float]], energy_type: str) -> List[Dict[str, Any]]:
        """Predict energy potential for multiple locations."""
        if energy_type not in self.models:
            raise ValueError(f"No trained model found for '{energy_type}'")
        
        predictions = []
        for i, (lat, lon, area) in enumerate(locations):
            try:
                prediction = self.predict_energy(lat, lon, energy_type, area)
                predictions.append(prediction)
                
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i + 1}/{len(locations)} locations")
                    
            except Exception as e:
                logging.error(f"Error predicting location {i}: {e}")
                predictions.append({'error': str(e), 'latitude': lat, 'longitude': lon})
        
        return predictions

# ------------------ 🚀 ENHANCED TEST SCRIPT ------------------
if __name__ == "__main__":
    predictor = EnergyPredictor()
    
    print("=" * 60)
    print("ENERGY PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Train all models
    for etype in ['solar', 'wind', 'hydro']:
        predictor.train_energy_model(etype, num_samples=500)
    
    # Save models
    predictor.save_models()
    
    print("\n" + "=" * 60)
    print("PREDICTION TESTING")
    print("=" * 60)
    
    # Test locations (city, lat, lon)
    test_locations = [
       
        ("Sahara Desert", 25.0, 0.0), 
        
    ]
    
    for city_name, lat, lon in test_locations:
        print(f"\n📍 {city_name} ({lat:.2f}, {lon:.2f})")
        print("-" * 40)
        
        analysis = predictor.analyze_location(lat, lon, area=100)
        
        for energy_type, prediction in analysis['predictions'].items():
            if 'error' not in prediction:
                print(f"  {energy_type.capitalize():6} → "
                      f"{prediction['predicted_energy_kwh']:>6.0f} kWh/year | "
                      f"Suitability: {prediction['suitability_score']:.2f} | "
                      f"Confidence: {prediction['confidence']:.2f}")
        
        if analysis['best_option']:
            best_type, best_pred = analysis['best_option']
            print(f" BEST: {best_type.upper()} "
                  f"({best_pred['suitability_score']:.2f} suitability)")
    
    print("\n" + "=" * 60)
    print("TRAINING AND PREDICTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Demonstrate batch prediction
    print("\n BATCH PREDICTION DEMO")
    batch_locations = [
        (37.7, -122.4, 100),
        (37.8, -122.5, 150),
        (37.6, -122.3, 200)
    ]
    
    batch_results = predictor.batch_predict(batch_locations, 'solar')
    print(f"Batch processed {len(batch_results)} locations")