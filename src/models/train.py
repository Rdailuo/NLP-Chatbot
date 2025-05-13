import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processing import preprocess_claim_narrative

def load_data(filepath: str) -> pd.DataFrame:
    """Load the insurance claims data."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for model training."""
    print("Preprocessing claim narratives...")
    
    # Extract features from claim narratives
    narrative_features = df['claim_narrative'].apply(preprocess_claim_narrative)
    features_df = pd.DataFrame(narrative_features.tolist())
    
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
    
    # Add categorical features (one-hot encoded)
    categorical_features = ['claim_type', 'policy_type', 'location']
    for feature in categorical_features:
        dummies = pd.get_dummies(df[feature], prefix=feature)
        features_df = pd.concat([features_df, dummies], axis=1)
    
    # Add numerical features
    for feature in numerical_features:
        features_df[feature] = df[feature]
    
    return features_df

def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train the fraud detection model."""
    print("Training model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Print model performance
    print("\nModel Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, scaler, feature_importance

def save_model(model, scaler, feature_importance, model_dir: str):
    """Save the trained model and related artifacts."""
    # Create model directory if it doesn't exist
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, 'fraud_detection_model.joblib')
    joblib.dump(model, model_path)
    
    # Save the scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Save feature importance
    importance_path = os.path.join(model_dir, 'feature_importance.json')
    feature_importance.to_json(importance_path, orient='records', indent=2)
    
    print(f"\nModel artifacts saved to {model_dir}")

def main():
    # Load the data
    data_path = 'data/raw/insurance_claims.csv'
    df = load_data(data_path)
    
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    # Prepare features
    X = prepare_features(df)
    y = df['is_fraudulent']
    
    # Train the model
    model, scaler, feature_importance = train_model(X, y)
    
    # Save the model and artifacts
    save_model(model, scaler, feature_importance, 'models/saved')

if __name__ == "__main__":
    main() 