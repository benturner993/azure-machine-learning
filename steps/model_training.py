"""
Step 2: Model Training
This is the second step in the Azure ML pipeline that trains a machine learning model
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from azure.ai.ml import MLClient, Input, Output
from azure.identity import DefaultAzureCredential

def train_model(input_data_path, scaler_path, model_output_path, metrics_output_path):
    """
    Train a machine learning model on the processed data
    
    Args:
        input_data_path (str): Path to processed data
        scaler_path (str): Path to fitted scaler
        model_output_path (str): Path to save trained model
        metrics_output_path (str): Path to save training metrics
    """
    print("Starting model training step...")
    
    # Load processed data
    print(f"Loading processed data from: {input_data_path}")
    df = pd.read_csv(input_data_path)
    
    # Load scaler
    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'target']
    X = df[feature_columns]
    y = df['target']
    
    print(f"Training data shape: {X.shape}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'n_features': len(feature_columns),
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'feature_importance_mean': float(np.mean(model.feature_importances_)),
        'feature_importance_std': float(np.std(model.feature_importances_))
    }
    
    # Save model and scaler together
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns
    }
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model_package, model_output_path)
    
    # Save metrics
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Model training completed!")
    print(f"Model package saved to: {model_output_path}")
    print(f"Metrics saved to: {metrics_output_path}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model_package, metrics

def main():
    """Main function for the model training step"""
    parser = argparse.ArgumentParser(description="Model Training Step")
    parser.add_argument("--input_data", type=str, help="Input processed data path")
    parser.add_argument("--scaler_path", type=str, help="Scaler path")
    parser.add_argument("--model_output", type=str, help="Model output path")
    parser.add_argument("--metrics_output", type=str, help="Metrics output path")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    input_data_path = args.input_data or "data/processed_data.csv"
    scaler_path = args.scaler_path or "models/scaler.pkl"
    model_output_path = args.model_output or "models/trained_model.pkl"
    metrics_output_path = args.metrics_output or "metrics/training_metrics.txt"
    
    # Train the model
    model_package, metrics = train_model(input_data_path, scaler_path, model_output_path, metrics_output_path)
    
    print("Model training step completed successfully!")

if __name__ == "__main__":
    main()
