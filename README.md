# 🚀 Azure ML Pipelines - Simple & Clean

Two separate, focused pipelines for Azure ML using SDK v2.

## 📋 What You Get

**🏋️ Training Pipeline** (`run-training.py`)
- Step 1: Data Processing (generates & processes data)
- Step 2: Model Training (trains & saves model)

**🔮 Inference Pipeline** (`run-inference.py`)  
- Step 1: Data Processing (processes inference data)
- Step 2: Inference & Model Registration (runs predictions & registers model)

## ⚡ Quick Start

### 1. Configure Azure Settings
Edit `config/settings.py` and update these 4 values:

```python
SUBSCRIPTION_ID = "your-subscription-id-here"      # Your Azure subscription ID
RESOURCE_GROUP = "your-resource-group-here"        # Resource group name
WORKSPACE_NAME = "your-workspace-name-here"        # Azure ML workspace name  
COMPUTE_NAME = "your-compute-cluster-name"         # Compute cluster name
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipelines
```bash
# Training Pipeline
python run-training.py

# Inference Pipeline  
python run-inference.py
```

**That's it!** 🎉

## 🔧 Where to Find Your Azure Settings

### Subscription ID
1. Go to [Azure Portal](https://portal.azure.com)
2. Search for "Subscriptions"
3. Copy your Subscription ID

### Resource Group & Workspace Name
1. Go to [Azure ML Studio](https://ml.azure.com)
2. Look at the top of the page - you'll see:
   - Resource Group: `your-resource-group-name`
   - Workspace: `your-workspace-name`

### Compute Cluster Name
1. In Azure ML Studio, go to "Compute" → "Compute clusters"
2. Use the name of your compute cluster
3. Don't have one? Create a new compute cluster first

## 📁 Project Structure

```
azure-ml/
├── config/
│   └── settings.py              # 🔧 Your configuration (edit this!)
├── steps/
│   ├── data_processing.py       # Data generation & processing
│   ├── model_training.py        # Model training  
│   └── inference.py             # Inference & model registration
├── run-training.py              # 🏋️ Training pipeline
├── run-inference.py             # 🔮 Inference pipeline
├── conda.yml                    # Environment dependencies
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎛️ Optional Settings

You can also customize these in `config/settings.py`:

```python
DATA_SAMPLES = 1000        # Number of data points to generate
DATA_FEATURES = 10         # Number of features per data point
MODEL_NAME = "my-ml-model" # Name for your model in Azure ML registry
```

## 🔍 What Happens When You Run

### Training Pipeline (`python run-training.py`)
1. **Data Processing**: Generates synthetic data using scikit-learn
2. **Model Training**: Trains a Random Forest model and saves it

### Inference Pipeline (`python run-inference.py`)  
1. **Data Processing**: Processes data for inference
2. **Inference**: Runs predictions and registers model to Azure ML

## 📊 Monitoring Your Pipelines

After running, check your pipelines at:
- **Azure ML Studio**: https://ml.azure.com
- Go to "Jobs" to see your pipeline runs
- Click on a job to see detailed logs and outputs

## 🛠️ Troubleshooting

### Configuration Check
```bash
python -c "from config.settings import print_status; print_status()"
```

### Common Issues

**❌ "Please update config/settings.py"**
- You haven't updated the 4 required values in `config/settings.py`

**❌ "Authentication failed"**  
- Run `az login` to authenticate with Azure

**❌ "Compute target not found"**
- Your compute cluster name is wrong or doesn't exist
- Create a compute cluster in Azure ML Studio first

**❌ "Workspace not found"**
- Check your subscription ID, resource group, and workspace name
- Make sure they match exactly what's in Azure

### Getting Help
1. Check the Azure ML Studio logs for detailed error messages
2. Verify all your settings in `config/settings.py`
3. Make sure your compute cluster is running

## 🎯 Customization

Want to modify the pipelines? Here's what to edit:

- **Data generation**: `steps/data_processing.py`
- **Model training**: `steps/model_training.py`  
- **Inference logic**: `steps/inference.py`
- **Pipeline structure**: `run-training.py` or `run-inference.py`
- **Settings**: `config/settings.py`

## 📋 Prerequisites

- Azure subscription with Azure ML workspace
- Python 3.8+
- Azure CLI installed (`az login` working)
- Compute cluster created in your Azure ML workspace

---

## 💡 Pro Tips

- **Run training first**: Always run the training pipeline before inference
- **Check logs**: Use Azure ML Studio to monitor your pipeline runs
- **Start small**: The default settings work great for testing
- **Scale up**: Increase `DATA_SAMPLES` for larger datasets

**Happy ML-ing!** 🚀