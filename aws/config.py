import os
import json
import yaml
import base64
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

@dataclass
class EC2Config:
    """Configuration for EC2 instances"""
    instance_type: str = "g4dn.xlarge"  # GPU instance for training
    ami_id: str = "ami-0c02fb55956c7d316"  # Ubuntu 20.04 LTS (update as needed)
    key_pair_name: str = "stranger-things-key"
    security_group_ids: List[str] = None
    subnet_id: str = ""
    region: str = "us-east-1"
    availability_zone: str = "us-east-1a"
    
    # Instance storage and compute
    volume_size: int = 100  # GB
    volume_type: str = "gp3"
    
    # Training specific settings
    max_spot_price: float = 0.50  # USD per hour for spot instances
    use_spot_instances: bool = True
    
    def __post_init__(self):
        if self.security_group_ids is None:
            self.security_group_ids = []

@dataclass
class S3Config:
    """Configuration for S3 storage"""
    bucket_name: str = "stranger-things-nlp"
    region: str = "us-east-1"
    
    # S3 paths structure
    training_data_prefix: str = "data/training/"
    model_prefix: str = "models/"
    results_prefix: str = "results/"
    logs_prefix: str = "logs/"
    checkpoints_prefix: str = "checkpoints/"

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Model settings
    base_model_llama: str = "meta-llama/Llama-3.2-3B-Instruct"
    base_model_qwen: str = "Qwen/Qwen2.5-3B-Instruct"
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_steps: int = 1000
    
    # Resource settings
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1

@dataclass
class GradioConfig:
    """Configuration for Gradio app deployment"""
    port: int = 7860
    host: str = "0.0.0.0"
    enable_auth: bool = True
    max_file_size: int = 50  # MB
    
    # SSL/Security (for production)
    enable_ssl: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""

class AWSConfigManager:
    """Manage AWS configurations for the project"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "aws_config.yaml"
        self.config_path = Path(self.config_file)
        
        # Default configurations
        self.ec2_config = EC2Config()
        self.s3_config = S3Config()
        self.training_config = TrainingConfig()
        self.gradio_config = GradioConfig()
        
        # Load existing config if available
        if self.config_path.exists():
            self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if 'ec2' in config_data:
                self.ec2_config = EC2Config(**config_data['ec2'])
            
            if 's3' in config_data:
                self.s3_config = S3Config(**config_data['s3'])
            
            if 'training' in config_data:
                self.training_config = TrainingConfig(**config_data['training'])
            
            if 'gradio' in config_data:
                self.gradio_config = GradioConfig(**config_data['gradio'])
                
            print(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
    
    def save_config(self):
        """Save current configuration to YAML file"""
        config_data = {
            'ec2': asdict(self.ec2_config),
            's3': asdict(self.s3_config),
            'training': asdict(self.training_config),
            'gradio': asdict(self.gradio_config)
        }
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            print(f"Saved configuration to {self.config_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_ec2_user_data_script(self) -> str:
        """Generate EC2 user data script for instance setup (Ubuntu)"""
        user_data = f"""#!/bin/bash
set -e

# Log all output for debugging
exec > >(tee /var/log/user-data.log) 2>&1
echo "Starting user data script at $(date)"

# Update system packages
apt update -y
apt upgrade -y

# Install essential packages for Ubuntu
apt install -y python3-pip python3-venv python3-dev git htop tree unzip curl wget
apt install -y build-essential software-properties-common

# Expand the root filesystem to use full EBS volume
echo "Expanding root filesystem..."
resize2fs /dev/$(df / | tail -1 | cut -d' ' -f1 | sed 's/[0-9]*$//')
echo "Root filesystem expanded"

# Set up additional EBS volumes
echo "Setting up additional storage volumes..."

# Wait for volumes to be available
sleep 10

# Format and mount model storage volume (handle both /dev/sdf and NVMe naming)
# Try NVMe naming first (for newer instances)
MODEL_DEVICE=""
CACHE_DEVICE=""

# Find available NVMe devices for model and cache storage
for device in /dev/nvme*n1; do
    if [ -b "$device" ] && [ "$device" != "/dev/nvme0n1" ]; then
        # Get device size to determine usage (200GB for models, 100GB for cache)
        SIZE=$(lsblk -b -n -d "$device" | awk '{{print $4}}')
        SIZE_GB=$((SIZE / 1024 / 1024 / 1024))
        
        if [ $SIZE_GB -ge 180 ] && [ -z "$MODEL_DEVICE" ]; then
            MODEL_DEVICE="$device"
            echo "Found model storage device: $device ($SIZE_GB GB)"
        elif [ $SIZE_GB -ge 80 ] && [ $SIZE_GB -lt 180 ] && [ -z "$CACHE_DEVICE" ]; then
            CACHE_DEVICE="$device"
            echo "Found cache storage device: $device ($SIZE_GB GB)"
        fi
    fi
done

# Fall back to traditional naming if NVMe not found
if [ -z "$MODEL_DEVICE" ] && [ -b "/dev/sdf" ]; then
    MODEL_DEVICE="/dev/sdf"
fi
if [ -z "$CACHE_DEVICE" ] && [ -b "/dev/sdg" ]; then
    CACHE_DEVICE="/dev/sdg"
fi

# Set up model storage volume
if [ -n "$MODEL_DEVICE" ] && [ -b "$MODEL_DEVICE" ]; then
    echo "Setting up model storage volume on $MODEL_DEVICE..."
    mkfs -t xfs "$MODEL_DEVICE"
    mkdir -p /mnt/models
    mount "$MODEL_DEVICE" /mnt/models
    echo "$MODEL_DEVICE /mnt/models xfs defaults,nofail 0 2" >> /etc/fstab
    chown ubuntu:ubuntu /mnt/models
    chmod 755 /mnt/models
    echo "Model storage volume mounted at /mnt/models"
else
    echo "Warning: No suitable device found for model storage"
fi

# Set up cache volume
if [ -n "$CACHE_DEVICE" ] && [ -b "$CACHE_DEVICE" ]; then
    echo "Setting up cache volume on $CACHE_DEVICE..."
    mkfs -t ext4 "$CACHE_DEVICE"
    mkdir -p /mnt/cache
    mount "$CACHE_DEVICE" /mnt/cache
    echo "$CACHE_DEVICE /mnt/cache ext4 defaults,nofail 0 2" >> /etc/fstab
    chown ubuntu:ubuntu /mnt/cache
    chmod 755 /mnt/cache
    echo "Cache volume mounted at /mnt/cache"
else
    echo "Warning: No suitable device found for cache storage"
fi

# Create subdirectories for organized storage
mkdir -p /mnt/models/{{checkpoints,trained_models,datasets}}
mkdir -p /mnt/cache/{{transformers,huggingface,pip}}
chown -R ubuntu:ubuntu /mnt/models /mnt/cache

echo "Additional volumes setup completed"

# Install NVIDIA drivers and CUDA (for GPU instances)
if lspci | grep -i nvidia > /dev/null; then
    echo "Installing NVIDIA drivers..."
    
    # Install NVIDIA drivers for Ubuntu
    apt install -y nvidia-driver-535 nvidia-utils-535
    
    # Install CUDA toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt update -y
    apt install -y cuda-toolkit-12-2
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda/bin${{PATH:+:${{PATH}}}}' >> /etc/environment
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}' >> /etc/environment
    
    echo "NVIDIA setup completed"
fi

# Create working directory
echo "Setting up working directory..."
mkdir -p /home/ubuntu/stranger-things-nlp
chown ubuntu:ubuntu /home/ubuntu/stranger-things-nlp

# Install AWS CLI v2 (if not already installed)
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI v2..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    ./aws/install
    rm -rf aws awscliv2.zip
fi

# Set up Python environment as ubuntu user
echo "Setting up Python environment..."
sudo -u ubuntu bash << 'EOF'
cd /home/ubuntu/stranger-things-nlp

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with CUDA support (latest compatible versions)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install commonly needed packages
echo "Installing basic ML packages..."
pip install transformers accelerate datasets peft bitsandbytes scipy numpy pandas

echo "Python environment setup completed" >> setup.log
echo "Setup completed at $(date)" >> setup.log
EOF

echo "User data script completed at $(date)"
"""
        return user_data
    
    def get_training_environment_vars(self) -> Dict[str, str]:
        """Get environment variables for training"""
        return {
            'CUDA_VISIBLE_DEVICES': '0',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
            'TRANSFORMERS_CACHE': '/mnt/cache/transformers',
            'HF_HOME': '/mnt/cache/huggingface',
            'HF_DATASETS_CACHE': '/mnt/cache/datasets',
            'TORCH_HOME': '/mnt/cache/torch',
            'WANDB_DISABLED': 'true',  # Disable wandb logging by default
            # Model storage paths
            'MODEL_STORAGE_PATH': '/mnt/models',
            'CHECKPOINT_PATH': '/mnt/models/checkpoints',
            'TRAINED_MODELS_PATH': '/mnt/models/trained_models',
            'DATASETS_PATH': '/mnt/models/datasets',
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate EC2 config
        if not self.ec2_config.key_pair_name:
            issues.append("EC2 key pair name is required")
        
        if not self.ec2_config.ami_id:
            issues.append("EC2 AMI ID is required")
        
        # Validate S3 config
        if not self.s3_config.bucket_name:
            issues.append("S3 bucket name is required")
        
        # Validate training config
        if self.training_config.batch_size < 1:
            issues.append("Batch size must be at least 1")
        
        if self.training_config.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        
        return issues
    
    def print_config_summary(self):
        """Print a summary of current configuration"""
        print("\n=== AWS Configuration Summary ===")
        print(f"EC2 Instance Type: {self.ec2_config.instance_type}")
        print(f"EC2 Region: {self.ec2_config.region}")
        print(f"Use Spot Instances: {self.ec2_config.use_spot_instances}")
        print(f"Max Spot Price: ${self.ec2_config.max_spot_price}/hour")
        print(f"S3 Bucket: {self.s3_config.bucket_name}")
        print(f"Training Base Model (LLaMA): {self.training_config.base_model_llama}")
        print(f"Training Base Model (Qwen): {self.training_config.base_model_qwen}")
        print(f"Gradio Port: {self.gradio_config.port}")
        print(f"Gradio Auth Enabled: {self.gradio_config.enable_auth}")
        
        issues = self.validate_config()
        if issues:
            print("\nConfiguration Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nConfiguration looks good!")

# Create default config file if it doesn't exist
def create_default_config():
    """Create default configuration file"""
    config_manager = AWSConfigManager()
    
    # Customize some defaults
    config_manager.s3_config.bucket_name = f"stranger-things-nlp-{os.getenv('USER', 'user')}"
    config_manager.ec2_config.key_pair_name = f"stranger-things-key-{os.getenv('USER', 'user')}"
    
    config_manager.save_config()
    config_manager.print_config_summary()
    
    return config_manager

if __name__ == "__main__":
    # Create default configuration
    config_manager = create_default_config()
    
    print(f"\nConfiguration saved to: {config_manager.config_path}")
    print("Edit the configuration file to customize settings for your AWS environment.")