#!/usr/bin/env python3
"""
Azure ML Training Pipeline
Two steps: Data Processing → Model Training
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Environment,
    PipelineJob,
    JobResourceConfiguration,
    UserIdentityConfiguration,
)
from azure.ai.ml import command, Output
from azure.identity import DefaultAzureCredential
import sys
import os
from datetime import datetime

# Import configuration
from config.settings import get_azure_config, is_configured, print_status, COMPUTE_NAME

def get_ml_client():
    """Get ML Client using Azure credentials"""
    if not is_configured():
        print("❌ Please update config/settings.py with your Azure details")
        sys.exit(1)
    
    config = get_azure_config()
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"],
    )

def create_environment():
    """Create environment for the pipeline"""
    return Environment(
        name="ml-training-env",
        description="Environment for ML training pipeline",
        image="mcr.microsoft.com/azureml/curated/sklearn-1.0-ubuntu20.04-py38-cpu:latest",
    )

def create_training_pipeline():
    """Create the training pipeline: Data Processing → Model Training"""
    
    # Create unique timestamp for job names
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Step 1: Data Processing
    data_step = command(
        name=f"data_processing_{timestamp}",
        command="python data_processing.py && echo 'data processing complete' > ${{outputs.completion_flag}}/done.txt",
        code="./steps/",
        compute=COMPUTE_NAME,
        environment=create_environment(),
        resources=JobResourceConfiguration(instance_count=1, instance_type="Standard_DS3_v2"),
        identity=UserIdentityConfiguration(),
        outputs={
            "completion_flag": Output(type="uri_folder")
        }
    )
    
    # Step 2: Model Training (runs after data processing)
    training_step = command(
        name=f"model_training_{timestamp}",
        command="python model_training.py",
        code="./steps/",
        compute=COMPUTE_NAME,
        environment=create_environment(),
        resources=JobResourceConfiguration(instance_count=1, instance_type="Standard_DS3_v2"),
        identity=UserIdentityConfiguration(),
        inputs={
            "wait_for_data": data_step.outputs.completion_flag
        }
    )
    
    # Create pipeline
    pipeline = PipelineJob(
        name=f"training-pipeline-{timestamp}",
        description="Training Pipeline: Data Processing → Model Training",
        jobs={
            "data_processing": data_step,
            "model_training": training_step
        },
        display_name=f"Training Pipeline {timestamp}",
    )
    
    return pipeline

def main():
    """Run the training pipeline"""
    print("🚀 Azure ML Training Pipeline")
    print("=" * 50)
    
    # Check configuration
    print_status()
    
    try:
        # Get ML client and create pipeline
        ml_client = get_ml_client()
        pipeline = create_training_pipeline()
        
        # Submit pipeline
        print("\n📤 Submitting training pipeline...")
        job = ml_client.jobs.create_or_update(pipeline)
        
        print(f"\n✅ Training pipeline submitted successfully!")
        print(f"🆔 Job ID: {job.name}")
        print(f"📊 Status: {job.status}")
        print(f"🌐 View in Azure ML Studio: https://ml.azure.com")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
