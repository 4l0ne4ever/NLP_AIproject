#!/usr/bin/env python3
"""
Test manual deployment to the running instance
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from aws.config import AWSConfigManager
from aws.ec2_orchestrator import EC2TrainingOrchestrator

def main():
    # Initialize config and orchestrator
    config = AWSConfigManager("aws_config.yaml")
    orchestrator = EC2TrainingOrchestrator(config)
    
    # Create instance info for the running instance
    instance_info = {
        'instance_id': 'i-01f1938d821bdd82d',
        'public_ip': '44.220.45.195',
        'private_ip': 'unknown',
        'status': 'running'
    }
    
    print(f"Attempting to deploy code to instance: {instance_info['public_ip']}")
    
    # Try to deploy code
    success = orchestrator.deploy_code_to_instance(instance_info)
    
    if success:
        print("Code deployment successful!")
        
        # Add to tracking
        job_name = "llama-training-manual"
        orchestrator.training_instances[job_name] = instance_info.copy()
        orchestrator.training_instances[job_name]['model_type'] = 'llama'
        orchestrator.training_instances[job_name]['launch_time'] = None
        
        print("Starting training...")
        training_success = orchestrator.start_training(job_name, "llama")
        
        if training_success:
            print("Training started successfully!")
            print(f"You can monitor the instance at: {instance_info['public_ip']}")
            print("SSH command: ssh -i ~/.ssh/stranger-things-key-dct.pem ec2-user@44.220.45.195")
            print("Check training logs: tail -f /home/ec2-user/stranger-things-nlp/training.log")
        else:
            print("Failed to start training")
    else:
        print("Code deployment failed")

if __name__ == "__main__":
    main()