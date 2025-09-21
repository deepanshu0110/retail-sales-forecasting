"""
🚀 STEP 5: DEPLOYMENT API
Production-ready Flask API for retail sales forecasting
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import logging
import pickle

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for loaded models and data
loaded_models = {}
model_metadata = {}
sample_data = None

# ---------------------------------------------------------------------
# Pre-flight: ensure JSON on POST routes (friendly 415 instead of 500)
# ---------------------------------------------------------------------
@app.before_request
def ensure_json_content_type():
    if request.method == 'POST' and request.path in ['/forecast', '/batch_forecast']:
        ctype = request.headers.get('Content-Type', '')
        if not ctype.startswith('application/json'):
            return jsonify({
                'error': 'Unsupported Media Type',
                'hint': "Set header: Content-Type: application/json",
                'path': request.path
            }), 415

def load_models():
    """Load trained models from disk"""
    global loaded_models, model_metadata
    try:
        models_dir = os.path.join(project_root, 'models')
        if not os.path.exists(models_dir):
            logger.warning("Models directory not found. Using fallback model.")
            return False

        # Load model metadata
        metadata_path = os.path.join(models_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)

        # Load available models
        for model_file in os.listdir(models_dir):
            if model_file.endswith('_model.pkl'):
                model_name = model_file.replace('_model.pkl', '')
                model_path = os.path.join(models_dir, model_file)
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        loaded_models[model_name] = model_data
                        logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {str(e)}")

        return len(loaded_models) > 0

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def load_sample_data():
    """Load sample data for fallback predictions"""
    global sample_data
    try:
        processed_path = os.path.join(project_root, 'data', 'processed', 'processed_sales_data.csv')
        if os.path.exists(processed_path):
            sample_data = pd.read_csv(processed_path)
            if 'date' in sample_data.columns:
                sample_data['date'] = pd.to_datetime(sample_data['date'])
            logger.info("Loaded processed data for fallback predictions")
            return True

        raw_path = os.path.join(project_root, 'data', 'raw', 'sales_data.csv')
        if os.path.exists(raw_path):
            sample_data = pd.read_csv(raw_path)
            if 'date' in sample_data.columns:
                sample_data['date'] = pd.to_datetime(sample_data['date'])
            logger.info("Loaded raw data for fallback predictions")
            return True

        logger.warning("No data files found. Using synthetic data.")
        return False

    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        return False

def generate_fallback_forecast(store_id=1, days_ahead=7):
    """Generate fallback forecast when models aren't available"""
    try:
        if sample_data is not None and {'store_id', 'sales'}.issubset(sample_data.columns):
            store_data = sample_data[sample_data['store_id'] == store_id].copy()
            if len(store_data) > 0:
                if 'day_of_week' not in store_data.columns:
                    if 'date' in store_data.columns and np.issubdtype(store_data['date'].dtype, np.datetime64):
                        store_data['day_of_week'] = store_data['date'].dt.weekday
                    else:
                        store_data['day_of_week'] = np.arange(len(store_data)) % 7

                tail_n = min(30, len(store_data))
                recent_avg = max(1e-6, store_data['sales'].tail(tail_n).mean())
                seasonal_pattern = store_data.groupby('day_of_week')['sales'].mean()

                forecasts = []
                for i in range(days_ahead):
                    dow = (datetime.now().weekday() + i) % 7
                    base = seasonal_pattern.get(dow, recent_avg)
                    seasonal_factor = base / recent_avg if recent_avg != 0 else 1.0
                    trend = i * 2
                    noise = np.random.normal(0, recent_avg * 0.05)
                    forecast = recent_avg * seasonal_factor + trend + noise
                    forecasts.append(max(0, float(forecast)))
                return forecasts

        # ultimate synthetic fallback
        base_sales = 1000 + (int(store_id) * 200)
        forecasts = []
        for i in range(days_ahead):
            dow = (datetime.now().weekday() + i) % 7
            weekend_boost = 1.2 if dow >= 5 else 1.0
            trend = i * 1.5
            noise = np.random.normal(0, 50)
            forecast = base_sales * weekend_boost + trend + noise
            forecasts.append(max(0, round(float(forecast), 2)))
        return forecasts

    except Exception as e:
        logger.error(f"Error generating fallback forecast: {str(e)}")
        return [max(0, float(1000 + np.random.normal(0, 100))) for _ in range(days_ahead)]

def home():
    return jsonify({
        'service': 'Retail Sales Forecasting API',
        'version': '1.0.0',
        'status': 'operational',
        'models_loaded': len(loaded_models),
        'endpoints': {
            '/forecast': 'POST - Generate sales forecast',
            '/batch_forecast': 'POST - Generate forecasts for multiple stores',
            '/health': 'GET - Health check',
            '/models': 'GET - List available models',
            '/': 'GET - This documentation'
        },
        'example_request': {
            'url': '/forecast',
            'method': 'POST',
            'body': {'store_id': 1, 'days_ahead': 7, 'model': 'random_forest'}
        }
    })

@app.route('/health', methods=['GET'])
def health():
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': len(loaded_models),
            'data_available': sample_data is not None,
            'uptime': 'running'
        }), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e), 'timestamp': datetime.now().isoformat()}), 500

@app.route('/models', methods=['GET'])
def list_models():
    try:
        models_info = {}
        for model_name, model_data in loaded_models.items():
            models_info[model_name] = {
                'description': model_data.get('description', 'No description'),
                'features': len(model_data.get('features', [])),
                'metrics': model_data.get('metrics', {}),
                'available': True
            }
        return jsonify({'models_count': len(loaded_models), 'models': models_info, 'metadata': model_metadata})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------ FIXED: use DataFrame to preserve feature names ------------------
def generate_model_forecast(model_data, store_id, days_ahead):
    """Generate forecast using a trained model, preserving feature names to avoid sklearn warnings"""
    try:
        model = model_data['model']
        features = model_data.get('features', [])

        if sample_data is not None and 'store_id' in sample_data.columns:
            store_data = sample_data[sample_data['store_id'] == store_id].tail(30)
            if len(store_data) > 0 and features:
                available_features = [f for f in features if f in store_data.columns]
                if not available_features:
                    return None
                recent_row = store_data[available_features].iloc[-1:].copy()   # 1-row DataFrame with column names
                forecast_df = pd.concat([recent_row] * days_ahead, ignore_index=True)  # repeat for horizon
                forecast_values = model.predict(forecast_df)  # keeps feature names, no warning
                return forecast_values.tolist()
        return None
    except Exception as e:
        logger.warning(f"Model forecast generation failed: {str(e)}")
        return None
# ------------------------------------------------------------------------------------

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        store_id = data.get('store_id', 1)
        days_ahead = data.get('days_ahead', 7)
        requested_model = data.get('model', 'best')

        if not isinstance(days_ahead, int) or days_ahead < 1 or days_ahead > 90:
            return jsonify({'error': 'days_ahead must be between 1 and 90'}), 400
        if not isinstance(store_id, int) or store_id < 1:
            return jsonify({'error': 'store_id must be a positive integer'}), 400

        start_date = datetime.now() + timedelta(days=1)
        forecast_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_ahead)]

        model_used = 'fallback'
        forecast_values = None
        confidence_interval = '80%'

        if loaded_models and requested_model != 'fallback':
            try:
                if requested_model == 'best':
                    best_model_name, best_rmse = None, float('inf')
                    for model_name, model_data in loaded_models.items():
                        rmse = model_data.get('metrics', {}).get('RMSE', float('inf'))
                        if rmse < best_rmse:
                            best_rmse, best_model_name = rmse, model_name
                    if best_model_name and 'model' in loaded_models[best_model_name]:
                        model_to_use = loaded_models[best_model_name]
                        model_used = best_model_name
                        forecast_values = generate_model_forecast(model_to_use, store_id, days_ahead)
                        confidence_interval = '85%'
                elif requested_model in loaded_models and 'model' in loaded_models[requested_model]:
                    model_to_use = loaded_models[requested_model]
                    model_used = requested_model
                    forecast_values = generate_model_forecast(model_to_use, store_id, days_ahead)
                    confidence_interval = '85%'
            except Exception as e:
                logger.warning(f"Model prediction failed: {str(e)}")

        if forecast_values is None:
            forecast_values = generate_fallback_forecast(store_id, days_ahead)
            model_used = 'fallback'
            confidence_interval = '70%'

        if len(forecast_values) != days_ahead:
            forecast_values = (
                forecast_values[:days_ahead] if len(forecast_values) > days_ahead
                else (forecast_values + [forecast_values[-1]] * (days_ahead - len(forecast_values))) if forecast_values
                else [0.0] * days_ahead
            )

        forecast_values = [max(0, round(float(v), 2)) for v in forecast_values]
        avg_forecast = float(np.mean(forecast_values)) if forecast_values else 0.0
        total_forecast = float(sum(forecast_values))

        response = {
            'store_id': store_id,
            'forecast_horizon': days_ahead,
            'forecast_dates': forecast_dates,
            'forecast_values': forecast_values,
            'model_used': model_used,
            'confidence_interval': confidence_interval,
            'summary': {
                'average_daily_forecast': round(avg_forecast, 2),
                'total_period_forecast': round(total_forecast, 2),
                'min_forecast': round(min(forecast_values), 2) if forecast_values else 0.0,
                'max_forecast': round(max(forecast_values), 2) if forecast_values else 0.0
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'api_version': '1.0.0',
                'forecast_type': 'daily_sales'
            }
        }

        if model_used in loaded_models:
            metrics = loaded_models[model_used].get('metrics', {})
            if metrics:
                response['model_performance'] = {
                    'rmse': round(metrics.get('RMSE', 0), 2),
                    'mape': round(metrics.get('MAPE', 0), 1),
                    'mae': round(metrics.get('MAE', 0), 2)
                }

        logger.info(f"Generated forecast for store {store_id}, {days_ahead} days using {model_used}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Forecast generation error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': str(e), 'timestamp': datetime.now().isoformat()}), 500

@app.route('/batch_forecast', methods=['POST'])
def batch_forecast():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        store_ids = data.get('store_ids', [1, 2, 3, 4, 5])
        days_ahead = data.get('days_ahead', 7)
        model = data.get('model', 'best')

        if not isinstance(store_ids, list) or len(store_ids) == 0 or len(store_ids) > 10:
            return jsonify({'error': 'store_ids must be a non-empty list with max 10 stores'}), 400
        if not all(isinstance(sid, int) and sid > 0 for sid in store_ids):
            return jsonify({'error': 'each store_id must be a positive integer'}), 400
        if not isinstance(days_ahead, int) or not (1 <= days_ahead <= 90):
            return jsonify({'error': 'days_ahead must be between 1 and 90'}), 400

        forecasts = {}
        for store_id in store_ids:
            try:
                forecast_values = None
                if loaded_models and model != 'fallback':
                    model_data = None
                    if model == 'best':
                        best_rmse, model_data = float('inf'), None
                        for model_name, model_info in loaded_models.items():
                            rmse = model_info.get('metrics', {}).get('RMSE', float('inf'))
                            if rmse < best_rmse and 'model' in model_info:
                                best_rmse, model_data = rmse, model_info
                    elif model in loaded_models and 'model' in loaded_models[model]:
                        model_data = loaded_models[model]
                    if model_data:
                        forecast_values = generate_model_forecast(model_data, store_id, days_ahead)

                if forecast_values is None:
                    forecast_values = generate_fallback_forecast(store_id, days_ahead)

                forecast_values = [max(0, round(float(v), 2)) for v in forecast_values[:days_ahead]]
                forecasts[f'store_{store_id}'] = {
                    'forecast_values': forecast_values,
                    'total_forecast': round(sum(forecast_values), 2),
                    'average_daily': round(float(np.mean(forecast_values)), 2) if forecast_values else 0.0
                }
            except Exception as e:
                forecasts[f'store_{store_id}'] = {'error': str(e), 'forecast_values': None}

        start_date = datetime.now() + timedelta(days=1)
        forecast_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_ahead)]

        return jsonify({
            'batch_forecast': True,
            'store_count': len(store_ids),
            'forecast_horizon': days_ahead,
            'forecast_dates': forecast_dates,
            'forecasts': forecasts,
            'model_used': model,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch forecast error: {str(e)}")
        return jsonify({'error': 'Batch forecast error', 'message': str(e)}), 500

# ------------------------- Initialization & Server -------------------------
def initialize_app():
    print("🚀 INITIALIZING FORECASTING API")
    print("=" * 40)

    models_loaded = load_models()
    print(f"✅ Loaded {len(loaded_models)} trained models" if models_loaded else "⚠️  No trained models found - using fallback predictions")

    data_loaded = load_sample_data()
    print("✅ Sample data loaded for enhanced predictions" if data_loaded else "⚠️  No sample data found - using synthetic fallback")

    print("\n🌐 API ENDPOINTS:")
    print("   GET  /                 - API documentation")
    print("   GET  /health           - Health check")
    print("   GET  /models           - List models")
    print("   POST /forecast         - Generate forecast")
    print("   POST /batch_forecast   - Batch forecasting")
    print("\n✅ API initialized successfully!")
    return models_loaded or data_loaded

if __name__ == '__main__':
    initialized = initialize_app()
    if not initialized:
        print("⚠️  API starting with limited functionality")

    print("\n🚀 STARTING FORECASTING API SERVER")
    print("=" * 45)
    print("📡 Server will be available at: http://localhost:5000")
    print("📖 API documentation at: http://localhost:5000")
    print("❤️  Health check at: http://localhost:5000/health")
    print("\n🛑 Press Ctrl+C to stop the server")

    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
