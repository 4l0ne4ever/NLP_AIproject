"""
AWS SageMaker Integration for Stranger Things NLP Project

This package provides comprehensive AWS SageMaker integration including:
- Training orchestration for scalable model training
- Model deployment and endpoint management
- S3-based storage and data management
- Production-ready Gradio web interface
- CLI tools for deployment and monitoring

Usage:
    # Setup infrastructure
    python sagemaker/deploy.py setup --bucket-name my-bucket

    # Train models
    python sagemaker/deploy.py train --model-type chatbot --data-path ./data

    # Deploy endpoints
    python sagemaker/deploy.py deploy --training-job-name job-name

    # Launch web interface
    python sagemaker/gradio_app.py

Key Components:
    - config: SageMaker configuration management
    - training_orchestrator: Training job management
    - deployment_manager: Model deployment and endpoints
    - storage: S3 integration and data management
    - gradio_app: SageMaker-powered web interface
    - deploy: Unified CLI for all operations
"""

from .config import SageMakerConfigManager, create_default_sagemaker_config
from .training_orchestrator import SageMakerTrainingOrchestrator, launch_chatbot_training
from .deployment_manager import SageMakerDeploymentManager, deploy_chatbot_model, quick_inference_test
from .storage import SageMakerS3Manager, create_sagemaker_bucket, upload_training_data

__version__ = "1.0.0"
__author__ = "Stranger Things NLP Project"

__all__ = [
    # Configuration
    "SageMakerConfigManager",
    "create_default_sagemaker_config",
    
    # Training
    "SageMakerTrainingOrchestrator", 
    "launch_chatbot_training",
    
    # Deployment
    "SageMakerDeploymentManager",
    "deploy_chatbot_model",
    "quick_inference_test",
    
    # Storage
    "SageMakerS3Manager",
    "create_sagemaker_bucket",
    "upload_training_data",
]