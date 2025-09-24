# SageMaker Integration for Stranger Things NLP Project

This directory contains a comprehensive AWS SageMaker integration that transforms your NLP project into a scalable, production-ready system. Instead of running models locally, you can now leverage SageMaker's managed infrastructure for training, deployment, and inference.

## Key Benefits of SageMaker Integration

- **Scalable Training**: Train models on powerful GPU instances without managing infrastructure
- **Auto-scaling Inference**: Endpoints that scale automatically based on demand
- **Cost Optimization**: Pay only for compute time used, with automatic shutdown
- **Production Ready**: Built-in monitoring, logging, and security
- **Easy Management**: Simple CLI and web interface for all operations
- **MLOps Integration**: Model versioning, A/B testing, and deployment pipelines

## What's Included

### Core Modules
- **`config.py`** - SageMaker-specific configuration management
- **`training_orchestrator.py`** - Launch and manage training jobs
- **`deployment_manager.py`** - Deploy and manage model endpoints
- **`storage.py`** - S3 integration for data and model management
- **`gradio_app.py`** - Web interface powered by SageMaker endpoints
- **`deploy.py`** - Unified CLI for all operations

## Quick Start Guide

### 1. Prerequisites

Before starting, ensure you have:

```bash
# AWS CLI configured with appropriate permissions
aws configure

# Required environment variables
export AWS_ACCOUNT_ID="123456789012"  # Your AWS account ID
export HUGGINGFACE_TOKEN="your_token_here"  # For model access

# Python dependencies
pip install -r ../requirements.txt
```

### 2. Infrastructure Setup

Initialize your SageMaker infrastructure:

```bash
# Set up S3 bucket and basic infrastructure
python deploy.py setup --bucket-name stranger-things-sagemaker-yourname

# This will:
# - Create S3 bucket with proper structure
# - Generate default configuration
# - Show next steps for IAM role creation
```

### 3. Create IAM Role

In AWS Console:
1. Go to IAM → Roles → Create Role
2. Select "SageMaker" as the service
3. Attach these policies:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
4. Name it `SageMakerExecutionRole`

### 4. Train Your First Model

```bash
# Upload training data and start training
python deploy.py train \
  --model-type chatbot \
  --data-path /path/to/your/training/data \
  --wait  # Optional: wait for completion

# This will:
# - Upload your data to S3
# - Create training job with GPU instances
# - Train character chatbot model
# - Save model artifacts to S3
```

### 5. Deploy Model to Endpoint

```bash
# Deploy the trained model
python deploy.py deploy \
  --training-job-name stranger-things-chatbot-123456 \
  --wait  # Optional: wait for endpoint to be ready

# This will:
# - Create SageMaker model from artifacts
# - Set up endpoint configuration
# - Launch auto-scaling endpoint
# - Enable monitoring and logging
```

### 6. Test Your Endpoint

```bash
# Test inference
python deploy.py test stranger-things-endpoint-123456 \
  --message "Hello Eleven, how are you?"

# Launch web interface
python gradio_app.py --port 7860
```

## Detailed Usage Guide

### Training Jobs

The training orchestrator supports multiple model types:

#### Character Chatbot Training
```bash
# Train Llama-based chatbot
python deploy.py train \
  --model-type chatbot \
  --data-path ./character_data/ \
  --job-name eleven-chatbot-v1

# Train with custom hyperparameters (advanced)
python -c "
from training_orchestrator import SageMakerTrainingOrchestrator
orchestrator = SageMakerTrainingOrchestrator()
job_arn = orchestrator.launch_training_job(
    job_name='custom-chatbot',
    model_type='chatbot',
    training_data_s3_uri='s3://bucket/data/',
    hyperparameters={
        'batch_size': '2',
        'learning_rate': '1e-4',
        'max_steps': '2000'
    }
)
"
```

#### Text Classification Training
```bash
# Train location classifier
python deploy.py train \
  --model-type text_classifier \
  --data-path ./classification_data/ \
  --job-name location-classifier-v1
```

### Model Deployment

#### Real-time Endpoints
```bash
# Deploy with auto-scaling
python -c "
from deployment_manager import SageMakerDeploymentManager
dm = SageMakerDeploymentManager()

# Deploy model
deployment_info = dm.deploy_model_complete(
    model_name='chatbot-model',
    model_artifacts_s3_uri='s3://bucket/models/model.tar.gz'
)

# Set up auto-scaling
dm.setup_auto_scaling(
    endpoint_name=deployment_info['endpoint_name'],
    min_capacity=1,
    max_capacity=10,
    target_value=100  # invocations per instance
)
"
```

#### Batch Processing
```bash
# Process large datasets
python -c "
from deployment_manager import SageMakerDeploymentManager
dm = SageMakerDeploymentManager()

job_arn = dm.create_batch_transform_job(
    job_name='batch-inference-job',
    model_name='deployed-model',
    input_s3_path='s3://bucket/batch-input/',
    output_s3_path='s3://bucket/batch-output/'
)
"
```

### Web Interface

The SageMaker-powered Gradio app provides a complete web interface:

```bash
# Launch with default settings
python gradio_app.py

# Launch with custom configuration
python gradio_app.py \
  --config custom_config.yaml \
  --port 8080 \
  --share  # Create public link
```

**Features:**
- **SageMaker Status**: Monitor endpoints and deployments
- **Character Chat**: Chat with Eleven using SageMaker endpoints
- **Text Classification**: Classify text using deployed models
- **Batch Processing**: Submit batch jobs through UI
- **Model Management**: Deploy new models from the interface

## Configuration Management

### Default Configuration

The system creates a `sagemaker_config.yaml` file with sensible defaults:

```yaml
training:
  instance_type: ml.g4dn.xlarge  # GPU for training
  max_runtime_seconds: 86400     # 24 hours
  hyperparameters:
    batch_size: '1'
    learning_rate: '2e-4'
    max_steps: '1000'

endpoint:
  instance_type: ml.g4dn.xlarge  # GPU for inference
  enable_auto_scaling: true
  min_capacity: 1
  max_capacity: 4

s3:
  bucket_name: stranger-things-sagemaker
  region: us-east-1
```

### Custom Configuration

Create custom configurations for different environments:

```python
from config import SageMakerConfigManager

# Load and modify configuration
config = SageMakerConfigManager('my_config.yaml')

# Update training settings
config.training_config.instance_type = 'ml.p3.2xlarge'  # More powerful GPU
config.training_config.max_steps = 2000

# Update endpoint settings
config.endpoint_config.enable_serverless = True  # Use serverless inference
config.endpoint_config.max_capacity = 10

# Save changes
config.save_config()
```

## Monitoring and Cost Management

### Built-in Monitoring

SageMaker provides built-in CloudWatch monitoring for:
- Training job metrics (loss, accuracy, resource utilization)
- Endpoint metrics (latency, throughput, errors)
- Auto-scaling events
- Cost tracking

### Cost Optimization Tips

1. **Use Spot Instances for Training**:
   ```python
   config.training_config.use_spot_instances = True
   config.training_config.max_spot_price = 0.50  # USD per hour
   ```

2. **Enable Auto-scaling**:
   ```python
   config.endpoint_config.enable_auto_scaling = True
   config.endpoint_config.min_capacity = 0  # Scale to zero when not used
   ```

3. **Use Batch Transform for Large Jobs**:
   - More cost-effective than real-time endpoints for batch processing
   - Automatically provisions and deprovisions resources

4. **Regular Cleanup**:
   ```bash
   # Clean up unused resources
   python deploy.py cleanup --keep-models --confirm
   ```

## Advanced Features

### Custom Training Scripts

The system generates optimized training scripts, but you can customize them:

```python
from training_orchestrator import SageMakerTrainingOrchestrator

orchestrator = SageMakerTrainingOrchestrator()

# Create custom training script
custom_script = orchestrator.create_training_script(
    model_type='chatbot',
    output_path='./custom_training_script.py'
)

# Modify the script as needed, then upload
# The system will use your custom script for training
```

### Multi-Model Endpoints

Deploy multiple models on a single endpoint for cost efficiency:

```python
from deployment_manager import SageMakerDeploymentManager

dm = SageMakerDeploymentManager()

# Configure multi-model endpoint
dm.config.endpoint_config.enable_multi_model = True

# Deploy with model switching capability
deployment_info = dm.deploy_model_complete(
    model_name='multi-model-endpoint',
    model_artifacts_s3_uri='s3://bucket/models/'
)
```

### A/B Testing

Deploy multiple model versions for comparison:

```python
# Deploy model A
endpoint_a = dm.create_endpoint('model-a-endpoint', 'model-a-config')

# Deploy model B
endpoint_b = dm.create_endpoint('model-b-endpoint', 'model-b-config')

# Route traffic between endpoints in your application
```

## Troubleshooting

### Common Issues

1. **IAM Permissions**:
   ```bash
   # Check your AWS permissions
   aws sts get-caller-identity
   aws iam get-role --role-name SageMakerExecutionRole
   ```

2. **Training Job Failures**:
   ```python
   # Check training job logs
   from training_orchestrator import SageMakerTrainingOrchestrator
   orchestrator = SageMakerTrainingOrchestrator()
   status = orchestrator.get_job_status('your-job-name')
   print(status['failure_reason'])
   ```

3. **Endpoint Issues**:
   ```python
   # Check endpoint status and logs
   from deployment_manager import SageMakerDeploymentManager
   dm = SageMakerDeploymentManager()
   status = dm.get_endpoint_status('your-endpoint')
   print(status)
   ```

4. **Resource Quotas**:
   - Check your SageMaker service quotas in AWS Console
   - Request limit increases if needed

### Getting Help

- Check CloudWatch logs for detailed error messages
- Use the `--debug` flag for verbose logging
- Monitor costs in AWS Cost Explorer
- Review SageMaker documentation for specific error codes

## Migration from EC2

If you're currently using the EC2 setup, here's how to migrate:

1. **Keep both systems running** during transition
2. **Export your trained models**:
   ```python
   # From EC2 system, upload models to S3
   from aws.storage import StrangerThingsS3Manager
   s3_manager = StrangerThingsS3Manager()
   s3_manager.upload_model('chatbot-v1', './local_model_path/')
   ```

3. **Deploy to SageMaker**:
   ```bash
   python deploy.py deploy --model-artifacts-uri s3://bucket/models/chatbot-v1/
   ```

4. **Test and compare** performance between systems
5. **Gradually shift traffic** to SageMaker endpoints
6. **Clean up EC2 resources** when confident in SageMaker setup

## Best Practices

1. **Use version control** for your configurations
2. **Test in development** before production deployment  
3. **Monitor costs regularly** using AWS Cost Explorer
4. **Set up CloudWatch alarms** for critical metrics
5. **Use separate environments** (dev/staging/prod)
6. **Regular model retraining** with new data
7. **Implement proper logging** and error handling
8. **Document your model versions** and deployment history

## Contributing

To extend the SageMaker integration:

1. **Add new model types** in `training_orchestrator.py`
2. **Create custom inference logic** in `deployment_manager.py`
3. **Extend the web interface** in `gradio_app.py`
4. **Add monitoring capabilities** for specific metrics
5. **Submit pull requests** with tests and documentation

## Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [HuggingFace on SageMaker](https://huggingface.co/docs/sagemaker/index)
- [Cost Optimization Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorchddp-saving-costs.html)

---

**Ready to bring your Stranger Things NLP project to production with AWS SageMaker!**

Start with `python deploy.py setup` and follow the quick start guide above.