"""
Step 3: Model Inference and Registration
This is the third step in the Azure ML pipeline that performs inference and registers the model
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import Model, ModelVersion
from azure.identity import DefaultAzureCredential
import sys
import os

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import get_azure_config, is_configured, MODEL_NAME, DATA_FEATURES

def perform_inference_and_register(processed_data_path, scaler_path, inference_output_path, model_name=None):
    """
    Perform inference on processed data and register the model to ML registry
    
    Args:
        processed_data_path (str): Path to processed data
        scaler_path (str): Path to scaler
        inference_output_path (str): Path to save inference results
        model_name (str): Name for model registration
    """
    print("Starting inference and model registration step...")
    
    # Use default model name if not provided
    if model_name is None:
        model_name = MODEL_NAME
    
    # Load processed data
    print(f"Loading processed data from: {processed_data_path}")
    processed_data_file = os.path.join(processed_data_path, "processed_data.csv")
    df = pd.read_csv(processed_data_file)
    
    # Load scaler
    print(f"Loading scaler from: {scaler_path}")
    scaler_file = os.path.join(scaler_path, "scaler.pkl")
    scaler = joblib.load(scaler_file)
    
    # Get feature columns (all except target)
    feature_columns = [col for col in df.columns if col != 'target']
    
    # Use the processed data for inference (simulate inference on new data)
    print("Using processed data for inference...")
    
    # Split the data - use part for inference
    feature_columns_final = [col for col in df.columns if col != 'target']
    X_inference = df[feature_columns_final].iloc[:200]  # Use first 200 rows for inference
    y_true = df['target'].iloc[:200]
    
    # For this demo, create a simple mock model for inference
    # In a real scenario, you would load your trained model here
    print("Creating mock model for inference demo...")
    from sklearn.ensemble import RandomForestClassifier
    mock_model = RandomForestClassifier(n_estimators=50, random_state=42)
    mock_model.fit(X_inference, y_true)  # Quick training for demo
    
    # Perform inference
    print("Performing inference...")
    y_pred = mock_model.predict(X_inference)
    y_pred_proba = mock_model.predict_proba(X_inference)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Create inference results
    inference_results = X_inference.copy()
    inference_results['true_class'] = y_true
    inference_results['predicted_class'] = y_pred
    inference_results['prediction_confidence'] = np.max(y_pred_proba, axis=1)
    inference_results['prediction_probability_class_0'] = y_pred_proba[:, 0]
    inference_results['prediction_probability_class_1'] = y_pred_proba[:, 1]
    
    # Save inference results
    os.makedirs(os.path.dirname(inference_output_path), exist_ok=True)
    inference_results.to_csv(inference_output_path, index=False)
    
    print(f"Inference completed. Results saved to: {inference_output_path}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Number of predictions: {len(y_pred)}")
    
    # Print classification report
    print("\nInference Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Register model to ML registry
    print(f"Registering model '{model_name}' to ML registry...")
    
    try:
        # Validate configuration
        if not is_configured():
            print("Warning: Azure configuration not set. Skipping model registration.")
            return inference_results, None
        
        # Get ML client
        config = get_azure_config()
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=config["subscription_id"],
            resource_group_name=config["resource_group"],
            workspace_name=config["workspace_name"],
        )
        
        # Save the mock model for registration
        model_path = os.path.join(inference_output_path, "model.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(mock_model, model_path)
        
        # Create model entity
        model_entity = Model(
            name=model_name,
            description=f"ML Pipeline Model - Accuracy: {accuracy:.4f}",
            path=model_path,
            type="mlflow_model",
            properties={
                "accuracy": str(accuracy),
                "n_features": str(len(feature_columns_final)),
                "model_type": "RandomForestClassifier"
            },
            tags={
                "pipeline": "ml-pipeline",
                "accuracy": str(accuracy)
            }
        )
        
        # Register the model
        registered_model = ml_client.models.create_or_update(model_entity)
        print(f"Model registered successfully!")
        print(f"Model ID: {registered_model.id}")
        print(f"Model Version: {registered_model.version}")
        
        return inference_results, registered_model
        
    except Exception as e:
        print(f"Error registering model: {str(e)}")
        print("Continuing without model registration...")
        return inference_results, None

def main():
    """Main function for the inference and model registration step"""
    parser = argparse.ArgumentParser(description="Inference and Model Registration Step")
    parser.add_argument("--processed_data", type=str, help="Path to processed data")
    parser.add_argument("--scaler_path", type=str, help="Path to scaler")
    parser.add_argument("--inference_output", type=str, help="Inference results output path")
    parser.add_argument("--model_name", type=str, help="Model name for registration")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    processed_data_path = args.processed_data or "data/processed_data.csv"
    scaler_path = args.scaler_path or "models/scaler.pkl"
    inference_output_path = args.inference_output or "outputs/inference_results.csv"
    model_name = args.model_name or MODEL_NAME
    
    # Perform inference and register model
    inference_results, registered_model = perform_inference_and_register(
        processed_data_path, scaler_path, inference_output_path, model_name
    )
    
    print("Inference and model registration step completed successfully!")

if __name__ == "__main__":
    main()
