#!/usr/bin/env python3
"""
SageMaker Deployment Script for Stranger Things NLP Project

This script provides a unified interface for deploying and managing your
SageMaker infrastructure including training jobs, endpoints, and monitoring.

Usage:
    python deploy.py setup --bucket-name my-sagemaker-bucket
    python deploy.py train --model-type chatbot --data-path ./data/
    python deploy.py deploy --training-job-name stranger-things-chatbot-123456
    python deploy.py status
    python deploy.py cleanup --keep-models
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import SageMakerConfigManager
from training_orchestrator import SageMakerTrainingOrchestrator
from deployment_manager import SageMakerDeploymentManager
from storage import SageMakerS3Manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SageMakerDeploymentCLI:
    """Command-line interface for SageMaker deployment operations"""
    
    def __init__(self, config_file: str = None):
        """Initialize deployment CLI"""
        self.config = SageMakerConfigManager(config_file)
        self.training_orchestrator = SageMakerTrainingOrchestrator(self.config)
        self.deployment_manager = SageMakerDeploymentManager(self.config)
        self.s3_manager = SageMakerS3Manager(
            bucket_name=self.config.s3_config.bucket_name,
            region=self.config.s3_config.region
        )
    
    def setup_infrastructure(self, bucket_name: str = None, create_role: bool = True):
        """Set up initial SageMaker infrastructure"""
        logger.info("Setting up SageMaker infrastructure...")
        
        if bucket_name:
            self.config.s3_config.bucket_name = bucket_name
            self.config._setup_defaults()
        
        try:
            # Test S3 connection and create bucket if needed
            logger.info(f"Setting up S3 bucket: {self.config.s3_config.bucket_name}")
            self.s3_manager = SageMakerS3Manager(
                bucket_name=self.config.s3_config.bucket_name,
                region=self.config.s3_config.region
            )
            
            # Save updated configuration
            self.config.save_config()
            
            # Show setup summary
            self.config.print_config_summary()
            
            # Create sample directories in S3
            sample_dirs = [
                "data/training/chatbot/",
                "data/training/text_classifier/",
                "models/",
                "checkpoints/",
                "results/"
            ]
            
            for dir_path in sample_dirs:
                # Create a placeholder file to establish directory structure
                placeholder_content = f"# Placeholder for {dir_path}\nCreated by SageMaker deployment script"
                self.s3_manager.s3_client.put_object(
                    Bucket=self.config.s3_config.bucket_name,
                    Key=f"{dir_path}.gitkeep",
                    Body=placeholder_content
                )
            
            logger.info("Infrastructure setup completed successfully!")
            
            if create_role:
                logger.info("\n💡 Next steps:")
                logger.info("1. Create SageMaker execution role in AWS Console:")
                logger.info("   - Go to IAM Console")
                logger.info("   - Create role with SageMaker service trust")
                logger.info("   - Attach AmazonSageMakerFullAccess policy")
                logger.info(f"   - Name it 'SageMakerExecutionRole'")
                logger.info("2. Update AWS account ID in environment:")
                logger.info("   export AWS_ACCOUNT_ID=your-account-id")
                logger.info("3. Ensure HuggingFace token is set:")
                logger.info("   export HUGGINGFACE_TOKEN=your-token")
            
            return True
            
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {e}")
            return False
    
    def train_model(self, model_type: str, data_path: str = None, 
                   job_name: str = None, wait: bool = False,
                   batch_size: int = None, learning_rate: float = None,
                   max_steps: int = None, num_epochs: int = None,
                   fp16: bool = None, base_model: str = None):
        """Launch a training job"""
        logger.info(f"Starting {model_type} model training...")
        
        try:
            # Prepare training data if local path provided
            if data_path and os.path.exists(data_path):
                logger.info(f"Uploading training data from {data_path}")
                training_data_uri = self.training_orchestrator.prepare_training_data(
                    data_path, model_type
                )
            else:
                # Use existing S3 data
                training_data_uri = self.config.get_s3_uri('training_data', model_type)
                logger.info(f"Using existing S3 data: {training_data_uri}")
            
            # Generate job name if not provided
            if not job_name:
                timestamp = int(time.time())
                job_name = f"stranger-things-{model_type}-{timestamp}"
            
            # Set model-specific hyperparameters
            hyperparameters = {}
            # Base model selection
            if model_type == "chatbot":
                # Use Llama as default for chatbot unless overridden
                hyperparameters['base_model'] = base_model or 'meta-llama/Llama-3.2-3B-Instruct'
            elif base_model:
                hyperparameters['base_model'] = base_model
            
            # Optional hyperparameters
            if batch_size is not None:
                hyperparameters['batch_size'] = batch_size
            if learning_rate is not None:
                hyperparameters['learning_rate'] = learning_rate
            if max_steps is not None:
                hyperparameters['max_steps'] = max_steps
            if num_epochs is not None:
                hyperparameters['num_epochs'] = num_epochs
            if fp16 is not None:
                hyperparameters['fp16'] = str(fp16).lower()
            
            # Launch training job
            job_arn = self.training_orchestrator.launch_training_job(
                job_name=job_name,
                model_type=model_type,
                training_data_s3_uri=training_data_uri,
                hyperparameters=hyperparameters
            )
            
            logger.info(f"Training job launched: {job_name}")
            logger.info(f"Job ARN: {job_arn}")
            
            # Wait for completion if requested
            if wait:
                logger.info("Waiting for training completion...")
                self._wait_for_training_job(job_name)
            
            return job_name
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def deploy_model(self, training_job_name: str = None, model_artifacts_uri: str = None,
                    endpoint_name: str = None, wait: bool = False):
        """Deploy a trained model to an endpoint"""
        logger.info("Deploying model to SageMaker endpoint...")
        
        try:
            if training_job_name:
                # Deploy from training job  
                deployment_info = self.deployment_manager.deploy_model_from_training_job(training_job_name)
            elif model_artifacts_uri:
                # Deploy from S3 artifacts
                model_name = endpoint_name or f"deployed-model-{int(time.time())}"
                deployment_info = self.deployment_manager.deploy_model_complete(
                    model_name=model_name,
                    model_artifacts_s3_uri=model_artifacts_uri,
                    endpoint_name=endpoint_name
                )
            else:
                logger.error("Either training_job_name or model_artifacts_uri must be provided")
                return None
            
            deployed_endpoint = deployment_info['endpoint_name']
            logger.info(f"Deployment initiated: {deployed_endpoint}")
            
            # Wait for endpoint to be ready if requested
            if wait:
                logger.info("Waiting for endpoint to be ready...")
                success = self.deployment_manager.wait_for_endpoint(deployed_endpoint)
                if success:
                    logger.info(f"Endpoint {deployed_endpoint} is ready for inference!")
                else:
                    logger.error(f"Endpoint {deployed_endpoint} failed to deploy")
            
            return deployed_endpoint
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return None
    
    def get_status(self):
        """Get status of all SageMaker resources"""
        logger.info("Getting SageMaker resource status...")
        
        try:
            # Training jobs
            training_jobs = self.training_orchestrator.list_training_jobs()
            print(f"\nRecent Training Jobs ({len(training_jobs)}):")
            for job in training_jobs[:5]:  # Show last 5
                status_text = job['status']
                print(f"  {job['job_name']}: {status_text}")
            
            # Endpoints
            endpoints = self.deployment_manager.list_active_endpoints()
            print(f"\nActive Endpoints ({len(endpoints)}):")
            for endpoint in endpoints:
                status_text = endpoint['status']
                print(f"  {endpoint['name']}: {status_text}")
            
            # S3 objects
            s3_objects = self.s3_manager.list_objects(max_keys=10)
            print(f"\nS3 Objects (showing first 10 of many):")
            for obj in s3_objects:
                size_mb = obj['Size'] / (1024 * 1024)
                print(f"  {obj['Key']}: {size_mb:.2f} MB")
            
            # Deployment summary
            summary = self.deployment_manager.get_deployment_summary()
            print(f"\nDeployment Summary:")
            print(f"  • Active Endpoints: {summary['endpoints']}")
            print(f"  • Registered Models: {summary['models']}")
            print(f"  • Batch Jobs: {summary['batch_jobs']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return False
    
    def cleanup_resources(self, keep_models: bool = True, keep_data: bool = True, 
                         confirm: bool = False):
        """Clean up SageMaker resources"""
        if not confirm:
            print("Warning: This will delete SageMaker resources!")
            print("Use --confirm flag to proceed with cleanup")
            return False
        
        logger.info("Cleaning up SageMaker resources...")
        
        try:
            cleanup_count = 0
            
            # Delete endpoints
            endpoints = self.deployment_manager.list_active_endpoints()
            for endpoint in endpoints:
                if endpoint['status'] in ['InService', 'Failed']:
                    logger.info(f"Deleting endpoint: {endpoint['name']}")
                    self.deployment_manager.delete_endpoint(
                        endpoint['name'], 
                        delete_config=True, 
                        delete_model=not keep_models
                    )
                    cleanup_count += 1
            
            # Clean up S3 artifacts if requested
            if not keep_data:
                logger.info("Cleaning up S3 training artifacts...")
                # This would delete checkpoints, logs, but keep final models if keep_models=True
                prefixes_to_clean = ['checkpoints/', 'logs/', 'data-capture/']
                if not keep_models:
                    prefixes_to_clean.append('models/')
                
                for prefix in prefixes_to_clean:
                    objects = self.s3_manager.list_objects(prefix)
                    for obj in objects:
                        self.s3_manager.delete_object(obj['Key'])
                        cleanup_count += 1
            
            logger.info(f"Cleanup completed. Removed {cleanup_count} resources.")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False
    
    def test_inference(self, endpoint_name: str, test_message: str = None):
        """Test inference on a deployed endpoint"""
        if not test_message:
            test_message = "Hello Eleven, how are you feeling today?"
        
        logger.info(f"Testing inference on endpoint: {endpoint_name}")
        
        try:
            # Test chatbot endpoint
            payload = {
                "inputs": test_message,
                "parameters": {
                    "max_length": 128,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            response = self.deployment_manager.invoke_endpoint(endpoint_name, payload)
            
            print(f"\nTest Results:")
            print(f"Input: {test_message}")
            print(f"Response: {response}")
            
            return True
            
        except Exception as e:
            logger.error(f"Inference test failed: {e}")
            return False
    
    def _wait_for_training_job(self, job_name: str, timeout: int = 7200):
        """Wait for training job to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status_info = self.training_orchestrator.get_job_status(job_name)
            status = status_info['status']
            
            if status == 'Completed':
                logger.info(f"Training job {job_name} completed successfully!")
                return True
            elif status == 'Failed':
                failure_reason = status_info.get('failure_reason', 'Unknown')
                logger.error(f"Training job {job_name} failed: {failure_reason}")
                return False
            elif status in ['InProgress', 'Starting']:
                logger.info(f"Training job status: {status}...")
                time.sleep(300)  # Check every 5 minutes
            else:
                logger.warning(f"Unexpected training status: {status}")
                time.sleep(60)
        
        logger.error(f"Timeout waiting for training job {job_name}")
        return False


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="SageMaker Deployment CLI for Stranger Things NLP")
    parser.add_argument("--config", help="SageMaker configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up SageMaker infrastructure")
    setup_parser.add_argument("--bucket-name", help="S3 bucket name for SageMaker")
    setup_parser.add_argument("--no-role-instructions", action="store_true", 
                             help="Skip IAM role setup instructions")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Launch training job")
    train_parser.add_argument("--model-type", choices=["chatbot", "text_classifier"], 
                             default="chatbot", help="Type of model to train")
    train_parser.add_argument("--data-path", help="Local path to training data")
    train_parser.add_argument("--job-name", help="Custom training job name")
    train_parser.add_argument("--wait", action="store_true", help="Wait for training completion")
    # Optional hyperparameters
    train_parser.add_argument("--batch-size", type=int, help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--max-steps", type=int, help="Max training steps")
    train_parser.add_argument("--num-epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--fp16", action="store_true", help="Enable fp16 training")
    train_parser.add_argument("--base-model", type=str, help="Base model name (e.g., meta-llama/Llama-3.2-3B-Instruct)")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model to endpoint")
    deploy_parser.add_argument("--training-job-name", help="Name of completed training job")
    deploy_parser.add_argument("--model-artifacts-uri", help="S3 URI of model artifacts")
    deploy_parser.add_argument("--endpoint-name", help="Custom endpoint name")
    deploy_parser.add_argument("--wait", action="store_true", help="Wait for endpoint to be ready")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get status of all resources")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test inference on endpoint")
    test_parser.add_argument("endpoint_name", help="Name of endpoint to test")
    test_parser.add_argument("--message", help="Custom test message")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up resources")
    cleanup_parser.add_argument("--keep-models", action="store_true", help="Keep trained models")
    cleanup_parser.add_argument("--keep-data", action="store_true", help="Keep training data")
    cleanup_parser.add_argument("--confirm", action="store_true", help="Confirm cleanup")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize CLI
    try:
        cli = SageMakerDeploymentCLI(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1
    
    # Execute commands
    if args.command == "setup":
        success = cli.setup_infrastructure(
            bucket_name=args.bucket_name,
            create_role=not args.no_role_instructions
        )
        return 0 if success else 1
    
    elif args.command == "train":
        job_name = cli.train_model(
            model_type=args.model_type,
            data_path=args.data_path,
            job_name=args.job_name,
            wait=args.wait,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            num_epochs=args.num_epochs,
            fp16=args.fp16,
            base_model=args.base_model
        )
        return 0 if job_name else 1
    
    elif args.command == "deploy":
        if not args.training_job_name and not args.model_artifacts_uri:
            logger.error("Either --training-job-name or --model-artifacts-uri is required")
            return 1
        
        endpoint = cli.deploy_model(
            training_job_name=args.training_job_name,
            model_artifacts_uri=args.model_artifacts_uri,
            endpoint_name=args.endpoint_name,
            wait=args.wait
        )
        return 0 if endpoint else 1
    
    elif args.command == "status":
        success = cli.get_status()
        return 0 if success else 1
    
    elif args.command == "test":
        success = cli.test_inference(args.endpoint_name, args.message)
        return 0 if success else 1
    
    elif args.command == "cleanup":
        success = cli.cleanup_resources(
            keep_models=args.keep_models,
            keep_data=args.keep_data,
            confirm=args.confirm
        )
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())