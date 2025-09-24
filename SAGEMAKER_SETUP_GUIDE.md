# 🚀 SageMaker Setup Guide - Stranger Things NLP Project

## Current Status ✅
Your environment is already well configured:
- ✅ AWS CLI configured (region: us-east-1)
- ✅ AWS Account ID: 159182460041
- ✅ HuggingFace token available
- ✅ SageMaker modules implemented
- ✅ Required dependencies installed

## Step-by-Step Setup Process

### Step 1: Create IAM Execution Role

First, you need to create a SageMaker execution role. Run this command to create the role:

```bash
# Create the IAM role
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document file://sagemaker-trust-policy.json
```

Then attach the required policies:

```bash
# Attach SageMaker full access
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Attach S3 full access
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Step 2: Initialize SageMaker Infrastructure

Run the setup command to create your S3 bucket and configure the infrastructure:

```bash
cd /Users/duongcongthuyet/Downloads/workspace/AI\ /project/sagemaker
python deploy.py setup --bucket-name stranger-things-sagemaker-159182460041
```

This will:
- Create your S3 bucket with the proper structure
- Set up directory structure for training data, models, and results
- Generate configuration files
- Show you the next steps

### Step 3: Verify Configuration

Check that your configuration is properly set up:

```bash
python -c "
from config import SageMakerConfigManager
config = SageMakerConfigManager()
config.print_config_summary()
"
```

### Step 4: Prepare Training Data

You have several options for training data:

#### Option A: Use existing character dialogue data
```bash
# If you have character data in CSV format
python deploy.py train --model-type chatbot --data-path ../character_chatbot/data/
```

#### Option B: Upload sample data to S3 for testing
```bash
# Create sample training data first
mkdir -p sample_data
echo "Hello Eleven, how are you today?" > sample_data/sample.txt
echo "I am doing well, thank you." >> sample_data/sample.txt

# Upload to S3
python deploy.py train --model-type chatbot --data-path ./sample_data/
```

### Step 5: Launch Your First Training Job

Once data is ready, launch a training job:

```bash
# Launch chatbot training (will take 30-60 minutes)
python deploy.py train \
  --model-type chatbot \
  --job-name stranger-things-eleven-test \
  --wait
```

### Step 6: Monitor Training Progress

While training is running, you can monitor progress:

```bash
# Check status
python deploy.py status

# Or monitor in AWS Console
# https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs
```

### Step 7: Deploy Model to Endpoint

After training completes successfully:

```bash
# Deploy the trained model
python deploy.py deploy \
  --training-job-name stranger-things-eleven-test \
  --wait
```

### Step 8: Test the Endpoint

Test your deployed model:

```bash
# Test inference
python deploy.py test \
  --endpoint-name stranger-things-endpoint-xxx \
  --message "Hello Eleven, how are you?"
```

### Step 9: Launch SageMaker-Enabled Gradio App

Finally, launch the web interface:

```bash
# Launch Gradio app with SageMaker endpoints
python gradio_app.py
```

## Alternative Quick Start (If You Want to Skip Training)

If you want to test the SageMaker infrastructure without waiting for training:

### Use Pre-trained Models from HuggingFace

```bash
# Deploy an existing model directly
python -c "
from deployment_manager import SageMakerDeploymentManager
from config import SageMakerConfigManager

config = SageMakerConfigManager()
dm = SageMakerDeploymentManager(config)

# Deploy a pre-trained model
deployment_info = dm.deploy_huggingface_model(
    model_id='microsoft/DialoGPT-medium',
    endpoint_name='stranger-things-test-endpoint'
)
print(f'Deployed endpoint: {deployment_info[\"endpoint_name\"]}')
"
```

## What Each Step Does

### Infrastructure Setup (`python deploy.py setup`)
- Creates S3 bucket: `stranger-things-sagemaker-159182460041`
- Sets up folder structure:
  - `data/training/chatbot/` - Training data
  - `models/` - Model artifacts  
  - `checkpoints/` - Training checkpoints
  - `results/` - Training results
- Configures IAM roles and permissions
- Generates `sagemaker_config.yaml`

### Training Job (`python deploy.py train`)
- Uploads your data to S3
- Creates SageMaker training job with GPU instances
- Uses custom Docker containers for PyTorch + HuggingFace
- Implements LoRA fine-tuning for efficient training
- Automatically saves model artifacts to S3
- Registers model in SageMaker Model Registry

### Model Deployment (`python deploy.py deploy`)
- Creates SageMaker model from training artifacts
- Sets up endpoint configuration with auto-scaling
- Deploys real-time inference endpoint
- Configures monitoring and logging
- Returns endpoint name for testing

### Gradio Interface (`python gradio_app.py`)
- Connects to SageMaker endpoints instead of local models
- Provides web interface for all NLP components
- Supports model switching between endpoints
- Real-time inference with SageMaker scaling

## Cost Estimation

### Training Costs (per job)
- **ml.g4dn.xlarge**: ~$1.20/hour
- **Typical training time**: 30-60 minutes
- **Cost per training job**: $0.60 - $1.20

### Inference Costs
- **ml.g4dn.xlarge endpoint**: ~$1.20/hour when active
- **Auto-scaling**: Scales down to 0 when not used
- **Alternative**: Use `ml.t2.medium` (~$0.05/hour) for testing

### Storage Costs
- **S3 storage**: ~$0.023/GB/month
- **Typical usage**: 1-5GB total = $0.02-$0.12/month

## Troubleshooting Common Issues

### 1. Role Creation Fails
```bash
# Check if role already exists
aws iam get-role --role-name SageMakerExecutionRole

# If it exists, you can skip role creation
```

### 2. Training Job Fails
```bash
# Check training job logs
python deploy.py logs --job-name your-job-name

# Common fixes:
# - Verify data format
# - Check S3 permissions
# - Ensure sufficient disk space
```

### 3. Endpoint Deployment Fails
```bash
# Check endpoint status
python deploy.py status

# Common fixes:
# - Verify model artifacts exist
# - Check instance capacity limits
# - Try different instance type
```

### 4. High Costs
```bash
# Clean up unused resources
python deploy.py cleanup --keep-models

# Stop all endpoints
python deploy.py stop-endpoints
```

## Next Steps After Setup

1. **Train Custom Models**: Use your own character dialogue data
2. **Implement Advanced Features**: 
   - Hyperparameter optimization
   - Multi-model endpoints  
   - A/B testing
3. **Production Deployment**:
   - Set up CI/CD pipelines
   - Configure monitoring alerts
   - Implement auto-scaling policies
4. **Cost Optimization**:
   - Use spot instances for training
   - Implement scheduled scaling
   - Set up cost alerts

## Support and Resources

- **AWS SageMaker Console**: https://console.aws.amazon.com/sagemaker/
- **Training Jobs**: Monitor progress and logs
- **Endpoints**: Manage deployments and scaling
- **Model Registry**: Version control for models

Your SageMaker setup is ready to go! Start with Step 1 and work through the process systematically.