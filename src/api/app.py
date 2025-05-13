from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processing import preprocess_claim_narrative

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def load_model():
    """Load the trained model and scaler."""
    global model, scaler
    
    try:
        model_path = os.path.join('models', 'saved', 'fraud_detection_model.joblib')
        scaler_path = os.path.join('models', 'saved', 'scaler.joblib')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        print("Model and scaler loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def prepare_features(claim_data: dict) -> pd.DataFrame:
    """Prepare features for prediction."""
    # Extract features from claim narrative
    narrative_features = preprocess_claim_narrative(claim_data['claim_narrative'])
    features_df = pd.DataFrame([narrative_features])
    
    # Add numerical features
    numerical_features = [
        'claim_amount',
        'policy_holder_age',
        'policy_holder_tenure',
        'previous_claims',
        'time_to_report',
        'deductible_amount',
        'coverage_amount'
    ]
    
    for feature in numerical_features:
        features_df[feature] = claim_data.get(feature, 0)
    
    # Add categorical features (one-hot encoded)
    categorical_features = ['claim_type', 'policy_type', 'location']
    for feature in categorical_features:
        value = claim_data.get(feature, '')
        features_df[f'{feature}_{value}'] = 1
    
    return features_df

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict fraud probability for a claim."""
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    try:
        # Get claim data from request
        claim_data = request.get_json()
        
        if not claim_data:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        # Prepare features
        features = prepare_features(claim_data)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        fraud_probability = model.predict_proba(features_scaled)[0][1]
        is_fraudulent = model.predict(features_scaled)[0]
        
        # Get feature importance for explanation
        feature_importance = pd.read_json(
            os.path.join('models', 'saved', 'feature_importance.json')
        )
        
        # Get top 3 most important features that contributed to the prediction
        top_features = feature_importance.head(3)['feature'].tolist()
        
        return jsonify({
            'claim_id': claim_data.get('claim_id', ''),
            'fraud_probability': float(fraud_probability),
            'is_fraudulent': bool(is_fraudulent),
            'prediction_confidence': float(max(fraud_probability, 1 - fraud_probability)),
            'top_contributing_features': top_features,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/model/metrics', methods=['GET'])
def get_model_metrics():
    """Get model performance metrics."""
    try:
        metrics_path = os.path.join('models', 'saved', 'model_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        else:
            return jsonify({
                'error': 'Model metrics not available'
            }), 404
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

def main():
    """Main function to run the Flask app."""
    # Load the model
    if not load_model():
        print("Failed to load model. Please train the model first.")
        return
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main() 