"""
Step 1: Data Generation and Processing
This is the first step in the Azure ML pipeline that generates and processes data using scikit-learn
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.settings import DATA_SAMPLES, DATA_FEATURES

def generate_and_process_data():
    """
    Generate synthetic data using scikit-learn and process it
    """
    print("Starting data generation and processing step...")
    
    # Generate synthetic classification data using scikit-learn
    print("Generating synthetic data using make_classification...")
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
    
    print(f"Generated data shape: {df.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Data preprocessing steps
    print("Performing data preprocessing...")
    
    # Separate features and target
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
    print("Performing feature engineering...")
    df_scaled['feature_1_squared'] = df_scaled['feature_1'] ** 2
    df_scaled['feature_interaction'] = df_scaled['feature_1'] * df_scaled['feature_2']
    df_scaled['feature_sum'] = df_scaled[['feature_1', 'feature_2', 'feature_3']].sum(axis=1)
    
    print(f"Data processing completed!")
    print(f"Processed data shape: {df_scaled.shape}")
    
    return df_scaled, scaler

def main():
    """Main function for the data generation and processing step"""
    # Generate and process the data
    processed_df, scaler = generate_and_process_data()
    
    print("Data generation and processing step completed successfully!")

if __name__ == "__main__":
    main()
