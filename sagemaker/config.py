"""
SageMaker Configuration for Stranger Things NLP Project

This module provides SageMaker-specific configuration management,
separate from the EC2-based AWS configuration.
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml
import time


@dataclass
class SageMakerTrainingConfig:
    """Configuration for SageMaker training jobs"""
    
    # Instance configuration
    instance_type: str = "ml.g4dn.xlarge"  # GPU instance for training
    instance_count: int = 1
    volume_size_gb: int = 100
    max_runtime_seconds: int = 86400  # 24 hours
    
    # Training job settings
    base_job_name: str = "stranger-things"
    role_arn: str = ""  # Will be set from environment or config
    
    # Hyperparameters (will be passed to training script)
    hyperparameters: Dict[str, Union[str, int, float]] = None
    
    # Input/Output configuration
    input_mode: str = "File"  # or "Pipe"
    output_path: str = ""  # S3 path for model artifacts
    
    # Training image configuration
    training_image: str = ""  # ECR URI for custom training image
    framework_version: str = "2.0.1"  # PyTorch version
    python_version: str = "py310"
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_s3_uri: str = ""
    
    # Distributed training
    enable_distributed_training: bool = False
    distribution_type: str = "data_parallel"  # or "model_parallel"
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {
                'batch_size': '1',
                'gradient_accumulation_steps': '4',
                'learning_rate': '2e-4',
                'num_epochs': '3',
                'max_steps': '1000',
                'lora_r': '64',
                'lora_alpha': '16',
                'lora_dropout': '0.1',
                'fp16': 'true',
                'gradient_checkpointing': 'true'
            }


@dataclass
class SageMakerEndpointConfig:
    """Configuration for SageMaker model endpoints"""
    
    # Endpoint configuration
    endpoint_name: str = ""
    endpoint_config_name: str = ""
    model_name: str = ""
    
    # Instance configuration
    instance_type: str = "ml.g4dn.xlarge"  # For real-time inference
    initial_instance_count: int = 1
    
    # Auto-scaling configuration
    enable_auto_scaling: bool = True
    min_capacity: int = 1
    max_capacity: int = 4
    target_invocations_per_instance: int = 100
    
    # Multi-model endpoint settings
    enable_multi_model: bool = True
    model_data_download_timeout: int = 600
    container_startup_health_check_timeout: int = 600
    
    # Serverless inference (alternative to real-time)
    enable_serverless: bool = False
    serverless_memory_size: int = 4096  # MB
    serverless_max_concurrency: int = 20


@dataclass
class SageMakerS3Config:
    """S3 configuration specific to SageMaker"""
    # Use a separate bucket from EC2 to avoid collisions
    bucket_name: str = "stranger-things-sagemaker-duongcongthuyet"
    region: str = "us-east-1"
    
    # SageMaker-specific S3 paths
    training_data_path: str = "data/training/"
    processed_data_path: str = "data/processed/"
    model_artifacts_path: str = "models/"
    checkpoints_path: str = "checkpoints/"
    logs_path: str = "logs/"
    batch_input_path: str = "batch-input/"
    batch_output_path: str = "batch-output/"
    data_capture_path: str = "data-capture/"
    pipeline_artifacts_path: str = "pipeline-artifacts/"


class SageMakerConfigManager:
    """Configuration manager for SageMaker components"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "sagemaker_config.yaml"
        self.config_path = Path(__file__).parent / self.config_file
        
        # Default configurations
        self.training_config = SageMakerTrainingConfig()
        self.endpoint_config = SageMakerEndpointConfig()
        self.s3_config = SageMakerS3Config()
        
        # Load existing config if available
        if self.config_path.exists():
            self.load_config()
        
        # Set up default values
        self._setup_defaults()
    
    def _setup_defaults(self):
        """Setup default values based on AWS environment"""
        # Get AWS configuration from environment
        region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        account_id = os.getenv('AWS_ACCOUNT_ID', '123456789012')
        
        # Set up S3 paths
        bucket_name = self.s3_config.bucket_name
        
        if not self.training_config.output_path:
            self.training_config.output_path = f"s3://{bucket_name}/{self.s3_config.model_artifacts_path}"
        
        if not self.training_config.checkpoint_s3_uri:
            self.training_config.checkpoint_s3_uri = f"s3://{bucket_name}/{self.s3_config.checkpoints_path}"
        
        # Set default role ARN
        if not self.training_config.role_arn:
            self.training_config.role_arn = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
        
        # Set training image URI to AWS PyTorch container
        if not self.training_config.training_image:
            self.training_config.training_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker"
        
        # Set default names with timestamps
        timestamp = int(time.time())
        
        if not self.endpoint_config.endpoint_name:
            self.endpoint_config.endpoint_name = f"stranger-things-endpoint-{timestamp}"
        
        if not self.endpoint_config.endpoint_config_name:
            self.endpoint_config.endpoint_config_name = f"stranger-things-config-{timestamp}"
        
        if not self.endpoint_config.model_name:
            self.endpoint_config.model_name = f"stranger-things-model-{timestamp}"
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if 'training' in config_data:
                self.training_config = SageMakerTrainingConfig(**config_data['training'])
            
            if 'endpoint' in config_data:
                self.endpoint_config = SageMakerEndpointConfig(**config_data['endpoint'])
            
            if 's3' in config_data:
                self.s3_config = SageMakerS3Config(**config_data['s3'])
            
            print(f"Loaded SageMaker configuration from {self.config_path}")
            
        except Exception as e:
            print(f"Error loading SageMaker config: {e}")
            print("Using default configuration")
    
    def save_config(self):
        """Save current configuration to YAML file"""
        config_data = {
            'training': asdict(self.training_config),
            'endpoint': asdict(self.endpoint_config),
            's3': asdict(self.s3_config)
        }
        
        try:
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            print(f"Saved SageMaker configuration to {self.config_path}")
        except Exception as e:
            print(f"❌ Error saving SageMaker config: {e}")
    
    def get_s3_uri(self, path_type: str, additional_path: str = "") -> str:
        """Get full S3 URI for different path types"""
        base_paths = {
            'training_data': self.s3_config.training_data_path,
            'processed_data': self.s3_config.processed_data_path,
            'models': self.s3_config.model_artifacts_path,
            'checkpoints': self.s3_config.checkpoints_path,
            'logs': self.s3_config.logs_path,
            'batch_input': self.s3_config.batch_input_path,
            'batch_output': self.s3_config.batch_output_path,
            'data_capture': self.s3_config.data_capture_path,
            'pipeline': self.s3_config.pipeline_artifacts_path
        }
        
        if path_type not in base_paths:
            raise ValueError(f"Unknown path type: {path_type}")
        
        base_path = base_paths[path_type]
        full_path = f"{base_path.rstrip('/')}/{additional_path}".rstrip('/')
        
        return f"s3://{self.s3_config.bucket_name}/{full_path}"
    
    def print_config_summary(self):
        """Print a summary of the SageMaker configuration"""
        print(f"\n=== SageMaker Configuration Summary ===")
        print(f"Training Instance: {self.training_config.instance_type}")
        print(f"Inference Instance: {self.endpoint_config.instance_type}")
        print(f"Auto-scaling: {'Yes' if self.endpoint_config.enable_auto_scaling else 'No'}")
        print(f"Multi-model Endpoint: {'Yes' if self.endpoint_config.enable_multi_model else 'No'}")
        print(f"Serverless: {'Yes' if self.endpoint_config.enable_serverless else 'No'}")
        print(f"S3 Bucket: {self.s3_config.bucket_name}")
        print(f"Model Output: {self.training_config.output_path}")
        print(f"Checkpoints: {self.training_config.checkpoint_s3_uri}")
        print("=" * 45)


# Create default configuration function
def create_default_sagemaker_config(custom_bucket_name: str = None) -> SageMakerConfigManager:
    """Create and save default SageMaker configuration"""
    config_manager = SageMakerConfigManager()
    
    if custom_bucket_name:
        config_manager.s3_config.bucket_name = custom_bucket_name
        # Update dependent paths
        config_manager._setup_defaults()
    
    config_manager.save_config()
    config_manager.print_config_summary()
    
    return config_manager


if __name__ == "__main__":
    # Interactive configuration setup
    print("🎬 Stranger Things SageMaker Configuration Setup")
    print("=" * 50)
    
    # Get user input for bucket name
    username = os.getenv('USER', 'user')
    default_bucket = f"stranger-things-sagemaker-{username}"
    
    print(f"Default S3 bucket name: {default_bucket}")
    custom_bucket = input("Enter custom bucket name (or press Enter for default): ").strip()
    
    if custom_bucket:
        bucket_name = custom_bucket
    else:
        bucket_name = default_bucket
    
    # Create configuration
    config_manager = create_default_sagemaker_config(bucket_name)
    
    print(f"\n✅ SageMaker configuration created!")
    print(f"📁 Config file: {config_manager.config_path}")
    print("\n💡 Next steps:")
    print("1. Review and edit the configuration file if needed")
    print("2. Ensure your AWS credentials are configured")
    print("3. Create the SageMaker execution role in your AWS account")
    print("4. Run the deployment script to set up your infrastructure")