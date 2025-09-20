"""
🔧 Azure ML Pipeline Configuration
Simply update the 4 values below with your Azure details
"""

# =============================================================================
# ✏️  EDIT THESE 4 VALUES - That's it!
# =============================================================================

SUBSCRIPTION_ID = "your-subscription-id-here"
RESOURCE_GROUP = "your-resource-group-here"  
WORKSPACE_NAME = "your-workspace-name-here"
COMPUTE_NAME = "your-compute-cluster-name"

# =============================================================================
# 🎛️  OPTIONAL SETTINGS (you can change these if you want)
# =============================================================================

# Data generation settings
DATA_SAMPLES = 1000        # How many data points to generate
DATA_FEATURES = 10         # How many features each data point has

# Model settings  
MODEL_NAME = "my-ml-model"  # Name for your model in Azure ML registry

# =============================================================================
# 🔧 HELPER FUNCTIONS (don't change these)
# =============================================================================

def get_azure_config():
    """Get your Azure configuration"""
    return {
        "subscription_id": SUBSCRIPTION_ID,
        "resource_group": RESOURCE_GROUP,
        "workspace_name": WORKSPACE_NAME
    }

def is_configured():
    """Check if you've updated the configuration"""
    placeholder_values = [
        "your-subscription-id-here",
        "your-resource-group-here", 
        "your-workspace-name-here",
        "your-compute-cluster-name"
    ]
    
    current_values = [SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, COMPUTE_NAME]
    
    # Return True if none of the values are still placeholders
    return not any(current in placeholders for current, placeholders in zip(current_values, placeholder_values))

def print_status():
    """Show current configuration status"""
    print("🔧 Configuration Status:")
    print("-" * 30)
    
    if is_configured():
        print("✅ Configuration looks good!")
        print(f"   📍 Workspace: {WORKSPACE_NAME}")
        print(f"   💻 Compute: {COMPUTE_NAME}")
        print(f"   🎯 Model Name: {MODEL_NAME}")
    else:
        print("❌ Configuration needed!")
        print("   Please update config/settings.py with your Azure details:")
        print("   • SUBSCRIPTION_ID")
        print("   • RESOURCE_GROUP")  
        print("   • WORKSPACE_NAME")
        print("   • COMPUTE_NAME")
        print()
        print("💡 Tip: You can find these values in the Azure portal")