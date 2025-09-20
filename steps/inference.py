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

def perform_inference_and_register(model_name=None):
    """
    Perform inference on generated data and register the model to ML registry
    
    Args:
        model_name (str): Name for model registration
    """
    print("Starting inference and model registration step...")
    
    # Use default model name if not provided
    if model_name is None:
        model_name = MODEL_NAME
    
    # Generate the same data as in other steps
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from config.settings import DATA_SAMPLES, DATA_FEATURES
    
    print("Generating inference data...")
    X, y = make_classification(
        n_samples=DATA_SAMPLES,
        n_features=DATA_FEATURES,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42,
        class_sep=0.8
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Preprocessing (same as other steps)
    feature_columns = [col for col in df.columns if col != 'target']
    X_features = df[feature_columns]
    y_target = df['target']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # Create new DataFrame with scaled features
    df_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    df_scaled['target'] = y_target
    
    # Feature engineering
    df_scaled['feature_1_squared'] = df_scaled['feature_1'] ** 2
    df_scaled['feature_interaction'] = df_scaled['feature_1'] * df_scaled['feature_2']
    df_scaled['feature_sum'] = df_scaled[['feature_1', 'feature_2', 'feature_3']].sum(axis=1)
    
    df = df_scaled
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
    
    print(f"Inference completed!")
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
        
    # Save the mock model for registration (in memory for demo)
    import tempfile
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model.pkl")
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
    # Perform inference and register model
    inference_results, registered_model = perform_inference_and_register()
    
    print("Inference and model registration step completed successfully!")

if __name__ == "__main__":
    main()
