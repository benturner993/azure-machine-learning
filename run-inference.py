#!/usr/bin/env python3
"""
Azure ML Inference Pipeline
Two steps: Data Processing → Inference & Model Registration
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
        name="ml-inference-env",
        description="Environment for ML inference pipeline",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file="conda.yml",
    )

def create_inference_pipeline():
    """Create the inference pipeline: Data Processing → Inference"""
    
    # Step 1: Data Processing (for inference data)
    data_step = command(
        name="data_processing",
        command="python data_processing.py --output_data ${{outputs.processed_data}} --scaler_output ${{outputs.scaler}}",
        code="./steps/",
        compute=COMPUTE_NAME,
        environment=create_environment(),
        resources=JobResourceConfiguration(instance_count=1, instance_type="Standard_DS3_v2"),
        identity=UserIdentityConfiguration(),
        outputs={
            "processed_data": Output(type="uri_folder", path="./data/"),
            "scaler": Output(type="uri_folder", path="./models/scaler/")
        }
    )
    
    # Step 2: Inference & Model Registration
    inference_step = command(
        name="inference",
        command="python inference.py --processed_data ${{inputs.processed_data}} --scaler_path ${{inputs.scaler_path}} --inference_output ${{outputs.inference_results}}",
        code="./steps/",
        compute=COMPUTE_NAME,
        environment=create_environment(),
        resources=JobResourceConfiguration(instance_count=1, instance_type="Standard_DS3_v2"),
        identity=UserIdentityConfiguration(),
        inputs={
            "processed_data": data_step.outputs.processed_data,
            "scaler_path": data_step.outputs.scaler
        },
        outputs={
            "inference_results": Output(type="uri_folder", path="./outputs/")
        }
    )
    
    # Create pipeline
    pipeline = PipelineJob(
        name="inference-pipeline",
        description="Inference Pipeline: Data Processing → Inference & Model Registration",
        jobs={
            "data_processing": data_step,
            "inference": inference_step
        },
        display_name="Inference Pipeline",
    )
    
    return pipeline

def main():
    """Run the inference pipeline"""
    print("🔮 Azure ML Inference Pipeline")
    print("=" * 50)
    
    # Check configuration
    print_status()
    
    try:
        # Get ML client and create pipeline
        ml_client = get_ml_client()
        pipeline = create_inference_pipeline()
        
        # Submit pipeline
        print("\n📤 Submitting inference pipeline...")
        job = ml_client.jobs.create_or_update(pipeline)
        
        print(f"\n✅ Inference pipeline submitted successfully!")
        print(f"🆔 Job ID: {job.name}")
        print(f"📊 Status: {job.status}")
        print(f"🌐 View in Azure ML Studio: https://ml.azure.com")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
