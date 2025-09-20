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

def train_model():
    """
    Train a machine learning model on generated data
    """
    print("Starting model training step...")
    
    # Generate the same data as in data processing step
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from config.settings import DATA_SAMPLES, DATA_FEATURES
    
    print("Generating training data...")
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
    
    # Preprocessing (same as data processing step)
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
    
    print(f"Model training completed!")
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, metrics

def main():
    """Main function for the model training step"""
    # Train the model
    model, metrics = train_model()
    
    print("Model training step completed successfully!")

if __name__ == "__main__":
    main()
