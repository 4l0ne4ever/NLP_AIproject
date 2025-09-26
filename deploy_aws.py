#!/usr/bin/env python3
"""
Stranger Things NLP Project - AWS Deployment CLI

This script provides easy commands to deploy and manage your project on AWS:
- Launch training instances on EC2
- Deploy and host Gradio app
- Manage S3 storage
- Monitor training jobs

Usage:
    python deploy_aws.py init                    # Initialize AWS config
    python deploy_aws.py train llama            # Start LLaMA training
    python deploy_aws.py train qwen             # Start Qwen training  
    python deploy_aws.py deploy-gradio          # Deploy Gradio app
    python deploy_aws.py list-instances         # List running instances
    python deploy_aws.py terminate <instance>   # Terminate instance
    python deploy_aws.py upload-data           # Upload training data to S3
    python deploy_aws.py status                 # Show overall status
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from aws.config import AWSConfigManager
from aws.storage import StrangerThingsS3Manager
from aws.ec2_orchestrator import EC2TrainingOrchestrator

class AWSDeploymentCLI:
    def __init__(self):
        self.config_manager = None
        self.s3_manager = None
        self.ec2_orchestrator = None
        self._init_managers()
    
    def _init_managers(self):
        """Initialize AWS managers"""
        try:
            self.config_manager = AWSConfigManager()
            self.s3_manager = StrangerThingsS3Manager(
                bucket_name=self.config_manager.s3_config.bucket_name
            )
            self.ec2_orchestrator = EC2TrainingOrchestrator(self.config_manager)
            print("AWS managers initialized successfully")
        except Exception as e:
            print(f"Error initializing AWS managers: {e}")
            print("Run 'python deploy_aws.py init' first to set up configuration")
            sys.exit(1)
    
    def init_config(self):
        """Initialize AWS configuration"""
        print("Initializing AWS configuration for Stranger Things NLP project...")
        
        from aws.config import create_default_config
        config_manager = create_default_config()
        
        print("\nNext steps:")
        print("1. Edit aws_config.yaml to customize your settings")
        print("2. Ensure your AWS credentials are configured (aws configure)")
        print("3. Create an EC2 key pair and update the key_pair_name in config")
        print("4. Run 'python deploy_aws.py upload-data' to upload training data")
        print("5. Run 'python deploy_aws.py train llama' to start training")
    
    def upload_data(self):
        """Upload training data to S3"""
        print("Uploading training data to S3...")
        
        # Check for local data directories
        data_dirs = [
            "data",
            "/content/data/transcripts/",  # Google Colab path
            "character_chatbot/result/",
            "evaluation_results.json",
            "evaluation_qwen_results.json"
        ]
        
        uploaded_something = False
        
        for data_dir in data_dirs:
            if Path(data_dir).exists():
                if Path(data_dir).is_dir():
                    print(f"Uploading directory: {data_dir}")
                    results = self.s3_manager.upload_training_data(data_dir)
                    success_count = sum(1 for success in results.values() if success)
                    print(f"Uploaded {success_count}/{len(results)} files from {data_dir}")
                    uploaded_something = True
                elif Path(data_dir).is_file():
                    print(f"Uploading file: {data_dir}")
                    s3_key = f"results/{Path(data_dir).name}"
                    success = self.s3_manager.upload_file(data_dir, s3_key)
                    if success:
                        print(f"Uploaded {data_dir}")
                        uploaded_something = True
        
        if not uploaded_something:
            print("No local data directories found. Expected directories:")
            for data_dir in data_dirs:
                print(f"  - {data_dir}")
    
    def start_training(self, model_type: str):
        """Start training on EC2"""
        print(f"Starting {model_type.upper()} training on EC2...")
        
        # Validate model type
        if model_type.lower() not in ['llama', 'qwen']:
            print("Model type must be 'llama' or 'qwen'")
            return
        
        try:
            # Launch training instance
            training_job_name = f"{model_type}-training-{int(__import__('time').time())}"
            instance_info = self.ec2_orchestrator.launch_training_instance(
                model_type=model_type.lower(),
                training_job_name=training_job_name
            )
            
            print(f"Training instance launched: {instance_info['public_ip']}")
            print(f"Job name: {training_job_name}")
            
            # Deploy code
            print("Deploying code to instance...")
            success = self.ec2_orchestrator.deploy_code_to_instance(instance_info)
            
            if success:
                print("Code deployed successfully")
                
                # Start training
                print("Starting training process...")
                training_started = self.ec2_orchestrator.start_training(
                    training_job_name,
                    model_type.lower(),
                    stream_logs=getattr(self, '_stream_logs', False),
                    log_lines=getattr(self, '_log_lines', 200),
                    s3_transcripts_prefix=getattr(self, '_s3_transcripts', None)
                )
                
                if training_started:
                    print(f"Training started successfully!")
                    print(f"SSH command: ssh -i ~/.ssh/{self.config_manager.ec2_config.key_pair_name}.pem ubuntu@{instance_info['public_ip']}")
                    print("Monitor training: tail -f training.log")
                else:
                    print("Failed to start training")
            else:
                print("Failed to deploy code")
                
        except Exception as e:
            print(f"Error starting training: {e}")
    
    def deploy_gradio(self):
        """Deploy Gradio app on EC2"""
        print("Deploying Gradio app on EC2...")
        
        try:
            # Launch Gradio instance
            instance_info = self.ec2_orchestrator.launch_gradio_instance()
            
            print(f"Gradio instance launched: {instance_info['public_ip']}")
            
            # Deploy code
            print("Deploying code to instance...")
            success = self.ec2_orchestrator.deploy_code_to_instance(instance_info)
            
            if success:
                print("Code deployed successfully")
                
                # Start Gradio app
                print("Starting Gradio application...")
                app_started = self.ec2_orchestrator.start_gradio_app()
                
                if app_started:
                    print("Gradio app started successfully!")
                    print(f"App URL: {instance_info['gradio_url']}")
                    print(f"SSH command: ssh -i ~/.ssh/{self.config_manager.ec2_config.key_pair_name}.pem ubuntu@{instance_info['public_ip']}")
                    print("Monitor app: tail -f gradio.log")
                    print("\nNote: It may take a few minutes for the app to be fully accessible")
                else:
                    print("Failed to start Gradio app")
            else:
                print("Failed to deploy code")
                
        except Exception as e:
            print(f"Error deploying Gradio app: {e}")
    
    def list_instances(self):
        """List all running instances"""
        print("Listing all project instances...")
        
        instances = self.ec2_orchestrator.list_instances()
        
        print("\nTraining Instances:")
        if instances['training']:
            for job_name, info in instances['training'].items():
                status = "ACTIVE" if info['status'] == 'training' else "IDLE"
                print(f"  [{status}] {job_name}: {info['public_ip']} ({info['status']})")
        else:
            print("  No training instances running")
        
        print("\nGradio Instances:")
        if instances['gradio']:
            for name, info in instances['gradio'].items():
                status = "ACTIVE" if info['status'] == 'running' else "IDLE"
                print(f"  [{status}] {name}: {info['gradio_url']} ({info['status']})")
        else:
            print("  No Gradio instances running")
    
    def terminate_instance(self, instance_id: str):
        """Terminate an EC2 instance"""
        print(f"Terminating instance: {instance_id}")
        
        try:
            success = self.ec2_orchestrator.terminate_instance(instance_id)
            if success:
                print("Instance terminated successfully")
            else:
                print("Failed to terminate instance")
        except Exception as e:
            print(f"Error terminating instance: {e}")
    
    def show_status(self):
        """Show overall project status"""
        print("Stranger Things NLP Project - AWS Status")
        print("=" * 50)
        
        # Configuration status
        print("\nConfiguration:")
        self.config_manager.print_config_summary()
        
        # S3 status
        print(f"\nS3 Storage (s3://{self.s3_manager.bucket_name}):")
        try:
            models = self.s3_manager.get_available_models()
            if models:
                print(f"  Available models: {', '.join(models)}")
            else:
                print("  No models found in S3")
            
            # List some objects
            objects = self.s3_manager.list_objects('')
            print(f"  Total objects in bucket: {len(objects)}")
            
        except Exception as e:
            print(f"  Error accessing S3: {e}")
        
        # EC2 status
        print("\nEC2 Instances:")
        self.list_instances()
        
        # Cost estimate
        print("\nEstimated Costs:")
        print(f"  • Training (g4dn.xlarge): ~${self.config_manager.ec2_config.max_spot_price}/hour (spot)")
        print(f"  • Gradio (t3.medium): ~$0.02/hour")
        print(f"  • S3 storage: ~$0.023/GB/month")
        print("\nTip: Remember to terminate instances when not in use to save costs!")

def main():
    parser = argparse.ArgumentParser(
        description="Stranger Things NLP - AWS Deployment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Initialize command
    subparsers.add_parser('init', help='Initialize AWS configuration')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Start model training')
    train_parser.add_argument('model', choices=['llama', 'qwen'], help='Model type to train')
    train_parser.add_argument('--stream-logs', action='store_true', help='Stream training logs to the terminal')
    train_parser.add_argument('--log-lines', type=int, default=200, help='Number of log lines to show when streaming (default: 200)')
    train_parser.add_argument('--s3-transcripts', type=str, help='S3 prefix for transcripts (e.g., data/training/transcripts/)')
    
    # Gradio deployment command
    subparsers.add_parser('deploy-gradio', help='Deploy Gradio app')
    
    # Instance management commands
    subparsers.add_parser('list-instances', help='List running instances')
    
    terminate_parser = subparsers.add_parser('terminate', help='Terminate an instance')
    terminate_parser.add_argument('instance_id', help='Instance ID to terminate')
    
    # Data management commands
    subparsers.add_parser('upload-data', help='Upload training data to S3')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show overall project status')
    status_parser.add_argument('--logs', action='store_true', help='Fetch and show recent training.log')
    status_parser.add_argument('--source', choices=['ssh', 's3'], default='s3', help='Where to fetch logs from (default: s3)')
    status_parser.add_argument('--model', choices=['llama', 'qwen'], help='Model type for S3 logs (if source=s3)')
    status_parser.add_argument('--job', help='Training job name (if source=ssh)')
    status_parser.add_argument('--lines', type=int, default=200, help='Number of log lines to show (default: 200)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Handle init command separately (doesn't need managers)
    if args.command == 'init':
        cli = AWSDeploymentCLI.__new__(AWSDeploymentCLI)  # Don't call __init__
        cli.init_config()
        return
    
    # Initialize CLI with managers
    try:
        cli = AWSDeploymentCLI()
    except SystemExit:
        return
    
    # Execute commands
    if args.command == 'train':
        # Store stream flags on cli to pass into orchestrator call
        cli._stream_logs = bool(getattr(args, 'stream_logs', False))
        cli._log_lines = int(getattr(args, 'log_lines', 200))
        cli._s3_transcripts = getattr(args, 's3_transcripts', None)
        cli.start_training(args.model)
    elif args.command == 'deploy-gradio':
        cli.deploy_gradio()
    elif args.command == 'list-instances':
        cli.list_instances()
    elif args.command == 'terminate':
        cli.terminate_instance(args.instance_id)
    elif args.command == 'upload-data':
        cli.upload_data()
    elif args.command == 'status':
        cli.show_status()
        # Optionally fetch logs
        if getattr(args, 'logs', False):
            print("\nRecent training.log:\n" + ("-"*50))
            content = cli.ec2_orchestrator.fetch_training_log(
                training_job_name=getattr(args, 'job', None),
                model_type=getattr(args, 'model', None),
                source=getattr(args, 'source', 's3'),
                lines=int(getattr(args, 'lines', 200))
            )
            print(content if content else "(no content)")

if __name__ == "__main__":
    main()