"""
SageMaker Deployment Manager for Stranger Things NLP Project

This module handles SageMaker model deployment including:
- Real-time endpoints for interactive inference
- Batch transform jobs for processing large datasets  
- Auto-scaling and multi-model endpoints
- Model versioning and A/B testing
"""

import boto3
import json
import time
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from botocore.exceptions import ClientError

from config import SageMakerConfigManager, SageMakerEndpointConfig
from storage import SageMakerS3Manager


class SageMakerDeploymentManager:
    """Manage SageMaker model deployments and endpoints"""
    
    def __init__(self, config: SageMakerConfigManager = None):
        self.config = config or SageMakerConfigManager()
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.config.s3_config.region)
        self.runtime_client = boto3.client('sagemaker-runtime', region_name=self.config.s3_config.region)
        self.s3_manager = SageMakerS3Manager(
            bucket_name=self.config.s3_config.bucket_name,
            region=self.config.s3_config.region
        )
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Track deployed resources
        self.active_endpoints = {}
        self.active_models = {}
        self.batch_jobs = {}
    
    def create_model(self, model_name: str, model_artifacts_s3_uri: str, 
                    execution_role_arn: str = None, 
                    inference_image: str = None) -> str:
        """
        Create a SageMaker model from training artifacts
        
        Args:
            model_name: Name for the model
            model_artifacts_s3_uri: S3 URI of model artifacts (model.tar.gz)
            execution_role_arn: IAM role for model execution
            inference_image: Docker image for inference (optional)
        
        Returns:
            Model ARN
        """
        
        # Set default execution role if not provided
        if not execution_role_arn:
            execution_role_arn = self.config.training_config.role_arn
        
        # Set default inference image based on model type
        if not inference_image:
            region = self.config.s3_config.region
            account_id = boto3.client('sts').get_caller_identity()['Account']
            # Use PyTorch inference container by default
            inference_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker"
        
        model_definition = {
            'ModelName': model_name,
            'PrimaryContainer': {
                'Image': inference_image,
                'ModelDataUrl': model_artifacts_s3_uri,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                    'SAGEMAKER_REGION': self.config.s3_config.region
                }
            },
            'ExecutionRoleArn': execution_role_arn,
            'Tags': [
                {'Key': 'Project', 'Value': 'stranger-things-nlp'},
                {'Key': 'Environment', 'Value': 'sagemaker'},
                {'Key': 'CreatedBy', 'Value': 'deployment-manager'}
            ]
        }
        
        try:
            response = self.sagemaker_client.create_model(**model_definition)
            
            # Track the model
            self.active_models[model_name] = {
                'arn': response['ModelArn'],
                'artifacts_uri': model_artifacts_s3_uri,
                'created_time': time.time(),
                'status': 'InService'
            }
            
            self.logger.info(f"✅ Created model: {model_name}")
            self.logger.info(f"📊 Model ARN: {response['ModelArn']}")
            
            return response['ModelArn']
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException' and 'already exists' in str(e):
                self.logger.warning(f"⚠️ Model {model_name} already exists")
                # Get existing model ARN
                try:
                    response = self.sagemaker_client.describe_model(ModelName=model_name)
                    return response['ModelArn']
                except ClientError:
                    pass
            
            self.logger.error(f"❌ Failed to create model: {e}")
            raise
    
    def create_endpoint_config(self, config_name: str, model_name: str,
                             instance_type: str = None,
                             instance_count: int = None) -> str:
        """
        Create SageMaker endpoint configuration
        
        Args:
            config_name: Name for the endpoint configuration
            model_name: Name of the model to deploy
            instance_type: EC2 instance type for hosting
            instance_count: Number of instances
        
        Returns:
            Endpoint configuration ARN
        """
        
        # Use defaults from config if not specified
        if not instance_type:
            instance_type = self.config.endpoint_config.instance_type
        if not instance_count:
            instance_count = self.config.endpoint_config.initial_instance_count
        
        config_definition = {
            'EndpointConfigName': config_name,
            'ProductionVariants': [
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': instance_count,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ],
            'Tags': [
                {'Key': 'Project', 'Value': 'stranger-things-nlp'},
                {'Key': 'Environment', 'Value': 'sagemaker'}
            ]
        }
        
        # Add data capture configuration if enabled
        if self.config.endpoint_config.enable_multi_model:
            config_definition['DataCaptureConfig'] = {
                'EnableCapture': True,
                'InitialSamplingPercentage': 10,
                'DestinationS3Uri': self.config.get_s3_uri('data_capture', config_name),
                'CaptureOptions': [
                    {'CaptureMode': 'Input'},
                    {'CaptureMode': 'Output'}
                ]
            }
        
        try:
            response = self.sagemaker_client.create_endpoint_config(**config_definition)
            
            self.logger.info(f"✅ Created endpoint config: {config_name}")
            return response['EndpointConfigArn']
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException' and 'already exists' in str(e):
                self.logger.warning(f"⚠️ Endpoint config {config_name} already exists")
                # Get existing config ARN
                try:
                    response = self.sagemaker_client.describe_endpoint_config(EndpointConfigName=config_name)
                    return response['EndpointConfigArn']
                except ClientError:
                    pass
            
            self.logger.error(f"❌ Failed to create endpoint config: {e}")
            raise
    
    def create_endpoint(self, endpoint_name: str, config_name: str) -> str:
        """
        Create SageMaker endpoint for real-time inference
        
        Args:
            endpoint_name: Name for the endpoint
            config_name: Name of the endpoint configuration
        
        Returns:
            Endpoint ARN
        """
        
        endpoint_definition = {
            'EndpointName': endpoint_name,
            'EndpointConfigName': config_name,
            'Tags': [
                {'Key': 'Project', 'Value': 'stranger-things-nlp'},
                {'Key': 'Environment', 'Value': 'sagemaker'},
                {'Key': 'Purpose', 'Value': 'real-time-inference'}
            ]
        }
        
        try:
            response = self.sagemaker_client.create_endpoint(**endpoint_definition)
            
            # Track the endpoint
            self.active_endpoints[endpoint_name] = {
                'arn': response['EndpointArn'],
                'config_name': config_name,
                'status': 'Creating',
                'created_time': time.time()
            }
            
            self.logger.info(f"🚀 Creating endpoint: {endpoint_name}")
            self.logger.info(f"📊 Endpoint ARN: {response['EndpointArn']}")
            self.logger.info("⏳ Endpoint creation can take 5-10 minutes...")
            
            return response['EndpointArn']
            
        except ClientError as e:
            self.logger.error(f"❌ Failed to create endpoint: {e}")
            raise
    
    def deploy_model_complete(self, model_name: str, model_artifacts_s3_uri: str,
                            endpoint_name: str = None) -> Dict[str, str]:
        """
        Complete model deployment: create model, config, and endpoint
        
        Args:
            model_name: Name for the model
            model_artifacts_s3_uri: S3 URI of model artifacts
            endpoint_name: Custom endpoint name (optional)
        
        Returns:
            Dictionary with ARNs of created resources
        """
        
        # Generate names if not provided
        if not endpoint_name:
            timestamp = int(time.time())
            endpoint_name = f"{model_name}-endpoint-{timestamp}"
        
        config_name = f"{model_name}-config-{int(time.time())}"
        
        self.logger.info(f"🚀 Starting complete deployment for {model_name}")
        
        try:
            # Step 1: Create model
            self.logger.info("📦 Step 1/3: Creating model...")
            model_arn = self.create_model(model_name, model_artifacts_s3_uri)
            
            # Step 2: Create endpoint configuration  
            self.logger.info("⚙️ Step 2/3: Creating endpoint configuration...")
            config_arn = self.create_endpoint_config(config_name, model_name)
            
            # Step 3: Create endpoint
            self.logger.info("🔗 Step 3/3: Creating endpoint...")
            endpoint_arn = self.create_endpoint(endpoint_name, config_name)
            
            deployment_info = {
                'model_name': model_name,
                'model_arn': model_arn,
                'config_name': config_name,
                'config_arn': config_arn,
                'endpoint_name': endpoint_name,
                'endpoint_arn': endpoint_arn,
                'status': 'Creating'
            }
            
            self.logger.info("✅ Deployment initiated successfully!")
            self.logger.info(f"🔍 Monitor endpoint status with: get_endpoint_status('{endpoint_name}')")
            
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"❌ Deployment failed: {e}")
            raise
    
    def get_endpoint_status(self, endpoint_name: str) -> Dict:
        """Get status of an endpoint"""
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            
            status_info = {
                'endpoint_name': endpoint_name,
                'status': response['EndpointStatus'],
                'creation_time': response['CreationTime'],
                'last_modified_time': response['LastModifiedTime'],
                'endpoint_arn': response['EndpointArn'],
                'endpoint_config_name': response['EndpointConfigName']
            }
            
            if response['EndpointStatus'] == 'Failed':
                status_info['failure_reason'] = response.get('FailureReason', 'Unknown')
            
            # Update local tracking
            if endpoint_name in self.active_endpoints:
                self.active_endpoints[endpoint_name]['status'] = response['EndpointStatus']
            
            return status_info
            
        except ClientError as e:
            self.logger.error(f"❌ Error getting endpoint status: {e}")
            return {'endpoint_name': endpoint_name, 'status': 'Unknown', 'error': str(e)}
    
    def wait_for_endpoint(self, endpoint_name: str, timeout: int = 1800) -> bool:
        """
        Wait for endpoint to be in service
        
        Args:
            endpoint_name: Name of endpoint to wait for
            timeout: Maximum wait time in seconds (default 30 minutes)
        
        Returns:
            True if endpoint is ready, False if timeout or failed
        """
        
        self.logger.info(f"⏳ Waiting for endpoint {endpoint_name} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status_info = self.get_endpoint_status(endpoint_name)
            status = status_info['status']
            
            if status == 'InService':
                self.logger.info(f"✅ Endpoint {endpoint_name} is ready!")
                return True
            elif status == 'Failed':
                failure_reason = status_info.get('failure_reason', 'Unknown')
                self.logger.error(f"❌ Endpoint {endpoint_name} failed: {failure_reason}")
                return False
            elif status in ['Creating', 'Updating']:
                self.logger.info(f"⏳ Endpoint status: {status}...")
                time.sleep(60)  # Check every minute
            else:
                self.logger.warning(f"⚠️ Unexpected endpoint status: {status}")
                time.sleep(30)
        
        self.logger.error(f"⏰ Timeout waiting for endpoint {endpoint_name}")
        return False
    
    def invoke_endpoint(self, endpoint_name: str, payload: Union[str, Dict], 
                       content_type: str = 'application/json') -> str:
        """
        Invoke endpoint for real-time inference
        
        Args:
            endpoint_name: Name of the endpoint
            payload: Input data for inference
            content_type: Content type of the payload
        
        Returns:
            Inference response
        """
        
        # Convert payload to string if it's a dictionary
        if isinstance(payload, dict):
            payload = json.dumps(payload)
        
        try:
            response = self.runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=content_type,
                Body=payload.encode('utf-8')
            )
            
            result = response['Body'].read().decode('utf-8')
            
            # Try to parse as JSON
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
                
        except ClientError as e:
            self.logger.error(f"❌ Error invoking endpoint: {e}")
            raise
    
    def create_batch_transform_job(self, job_name: str, model_name: str,
                                 input_s3_path: str, output_s3_path: str,
                                 instance_type: str = None,
                                 instance_count: int = 1) -> str:
        """
        Create batch transform job for processing large datasets
        
        Args:
            job_name: Name for the batch job
            model_name: Name of the model to use
            input_s3_path: S3 path with input data
            output_s3_path: S3 path for output results
            instance_type: EC2 instance type for batch processing
            instance_count: Number of instances
        
        Returns:
            Transform job ARN
        """
        
        if not instance_type:
            instance_type = "ml.m5.xlarge"  # CPU instance for batch processing
        
        job_definition = {
            'TransformJobName': job_name,
            'ModelName': model_name,
            'TransformInput': {
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': input_s3_path
                    }
                },
                'ContentType': 'application/json',
                'SplitType': 'Line'
            },
            'TransformOutput': {
                'S3OutputPath': output_s3_path,
                'Accept': 'application/json'
            },
            'TransformResources': {
                'InstanceType': instance_type,
                'InstanceCount': instance_count
            },
            'Tags': [
                {'Key': 'Project', 'Value': 'stranger-things-nlp'},
                {'Key': 'Environment', 'Value': 'sagemaker'},
                {'Key': 'Purpose', 'Value': 'batch-processing'}
            ]
        }
        
        try:
            response = self.sagemaker_client.create_transform_job(**job_definition)
            
            # Track the batch job
            self.batch_jobs[job_name] = {
                'arn': response['TransformJobArn'],
                'model_name': model_name,
                'input_path': input_s3_path,
                'output_path': output_s3_path,
                'status': 'InProgress',
                'created_time': time.time()
            }
            
            self.logger.info(f"🔄 Started batch transform job: {job_name}")
            self.logger.info(f"📊 Job ARN: {response['TransformJobArn']}")
            
            return response['TransformJobArn']
            
        except ClientError as e:
            self.logger.error(f"❌ Failed to create batch transform job: {e}")
            raise
    
    def get_batch_job_status(self, job_name: str) -> Dict:
        """Get status of batch transform job"""
        try:
            response = self.sagemaker_client.describe_transform_job(TransformJobName=job_name)
            
            status_info = {
                'job_name': job_name,
                'status': response['TransformJobStatus'],
                'creation_time': response['CreationTime'],
                'transform_start_time': response.get('TransformStartTime'),
                'transform_end_time': response.get('TransformEndTime'),
                'failure_reason': response.get('FailureReason')
            }
            
            # Update local tracking
            if job_name in self.batch_jobs:
                self.batch_jobs[job_name]['status'] = response['TransformJobStatus']
            
            return status_info
            
        except ClientError as e:
            self.logger.error(f"❌ Error getting batch job status: {e}")
            return {'job_name': job_name, 'status': 'Unknown', 'error': str(e)}
    
    def setup_auto_scaling(self, endpoint_name: str, variant_name: str = 'primary',
                          min_capacity: int = None, max_capacity: int = None,
                          target_metric: str = 'InvocationsPerInstance',
                          target_value: int = None) -> bool:
        """
        Set up auto-scaling for an endpoint
        
        Args:
            endpoint_name: Name of the endpoint
            variant_name: Production variant name
            min_capacity: Minimum instance count
            max_capacity: Maximum instance count
            target_metric: Metric to scale on
            target_value: Target value for the metric
        
        Returns:
            Success status
        """
        
        # Use defaults from config if not specified
        if not min_capacity:
            min_capacity = self.config.endpoint_config.min_capacity
        if not max_capacity:
            max_capacity = self.config.endpoint_config.max_capacity
        if not target_value:
            target_value = self.config.endpoint_config.target_invocations_per_instance
        
        try:
            # Create application auto scaling client
            autoscaling_client = boto3.client('application-autoscaling', 
                                            region_name=self.config.s3_config.region)
            
            # Register scalable target
            resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
            
            autoscaling_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=min_capacity,
                MaxCapacity=max_capacity
            )
            
            # Create scaling policy
            policy_name = f"{endpoint_name}-scaling-policy"
            
            autoscaling_client.put_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': float(target_value),
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': f'SageMaker{target_metric}'
                    },
                    'ScaleOutCooldown': 300,  # 5 minutes
                    'ScaleInCooldown': 300    # 5 minutes
                }
            )
            
            self.logger.info(f"✅ Auto-scaling configured for endpoint {endpoint_name}")
            self.logger.info(f"📊 Min: {min_capacity}, Max: {max_capacity}, Target: {target_value}")
            
            return True
            
        except ClientError as e:
            self.logger.error(f"❌ Failed to set up auto-scaling: {e}")
            return False
    
    def delete_endpoint(self, endpoint_name: str, delete_config: bool = True,
                       delete_model: bool = False) -> bool:
        """
        Delete endpoint and optionally associated resources
        
        Args:
            endpoint_name: Name of endpoint to delete
            delete_config: Whether to delete endpoint configuration
            delete_model: Whether to delete the model
        
        Returns:
            Success status
        """
        
        try:
            # Get endpoint details before deletion
            endpoint_info = self.get_endpoint_status(endpoint_name)
            config_name = endpoint_info.get('endpoint_config_name')
            
            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            self.logger.info(f"🗑️ Deleted endpoint: {endpoint_name}")
            
            # Delete endpoint configuration if requested
            if delete_config and config_name:
                try:
                    self.sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
                    self.logger.info(f"🗑️ Deleted endpoint config: {config_name}")
                except ClientError as e:
                    self.logger.warning(f"⚠️ Could not delete config {config_name}: {e}")
            
            # Delete model if requested
            if delete_model:
                # Extract model name from config (would need to describe config first)
                # This is a simplified version
                model_name = endpoint_name.replace('-endpoint', '').split('-')[0]
                try:
                    self.sagemaker_client.delete_model(ModelName=model_name)
                    self.logger.info(f"🗑️ Deleted model: {model_name}")
                except ClientError as e:
                    self.logger.warning(f"⚠️ Could not delete model {model_name}: {e}")
            
            # Remove from tracking
            if endpoint_name in self.active_endpoints:
                del self.active_endpoints[endpoint_name]
            
            return True
            
        except ClientError as e:
            self.logger.error(f"❌ Error deleting endpoint: {e}")
            return False
    
    def list_active_endpoints(self) -> List[Dict]:
        """List all active endpoints for the project"""
        try:
            response = self.sagemaker_client.list_endpoints(
                SortBy='CreationTime',
                SortOrder='Descending',
                MaxResults=50
            )
            
            project_endpoints = []
            for endpoint in response['Endpoints']:
                # Filter for project endpoints
                if 'stranger-things' in endpoint['EndpointName'].lower():
                    endpoint_info = {
                        'name': endpoint['EndpointName'],
                        'status': endpoint['EndpointStatus'],
                        'creation_time': endpoint['CreationTime'],
                        'last_modified_time': endpoint['LastModifiedTime']
                    }
                    project_endpoints.append(endpoint_info)
            
            return project_endpoints
            
        except ClientError as e:
            self.logger.error(f"❌ Error listing endpoints: {e}")
            return []
    
    def get_deployment_summary(self) -> Dict:
        """Get summary of all deployments"""
        summary = {
            'endpoints': len(self.active_endpoints),
            'models': len(self.active_models),
            'batch_jobs': len(self.batch_jobs),
            'active_endpoints': list(self.active_endpoints.keys()),
            'active_models': list(self.active_models.keys()),
            'active_batch_jobs': list(self.batch_jobs.keys())
        }
        
        return summary


# Convenience functions for quick deployments
def deploy_chatbot_model(training_job_name: str, 
                        config: SageMakerConfigManager = None) -> Dict[str, str]:
    """Quick deploy a trained chatbot model"""
    
    deployment_manager = SageMakerDeploymentManager(config)
    
    # Get model artifacts from completed training job
    model_artifacts_uri = deployment_manager.s3_manager.get_model_artifacts(training_job_name)
    
    if not model_artifacts_uri:
        raise ValueError(f"No model artifacts found for training job: {training_job_name}")
    
    # Deploy the model
    model_name = f"chatbot-{training_job_name}"
    return deployment_manager.deploy_model_complete(model_name, model_artifacts_uri)


def quick_inference_test(endpoint_name: str, test_message: str, 
                        config: SageMakerConfigManager = None) -> str:
    """Quick test inference on deployed chatbot endpoint"""
    
    deployment_manager = SageMakerDeploymentManager(config)
    
    # Prepare payload for chatbot inference
    payload = {
        "inputs": test_message,
        "parameters": {
            "max_length": 256,
            "temperature": 0.7,
            "do_sample": True
        }
    }
    
    return deployment_manager.invoke_endpoint(endpoint_name, payload)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="SageMaker Deployment Manager")
    parser.add_argument("--action", choices=["deploy", "status", "list", "test", "delete"], required=True)
    parser.add_argument("--model-artifacts-uri", help="S3 URI of model artifacts")
    parser.add_argument("--model-name", help="Model name")
    parser.add_argument("--endpoint-name", help="Endpoint name")
    parser.add_argument("--test-message", help="Test message for inference")
    
    args = parser.parse_args()
    
    # Create deployment manager
    deployment_manager = SageMakerDeploymentManager()
    
    if args.action == "deploy" and args.model_name and args.model_artifacts_uri:
        result = deployment_manager.deploy_model_complete(args.model_name, args.model_artifacts_uri)
        print(f"✅ Deployment initiated: {json.dumps(result, indent=2, default=str)}")
    
    elif args.action == "status" and args.endpoint_name:
        status = deployment_manager.get_endpoint_status(args.endpoint_name)
        print(f"📊 Endpoint Status: {json.dumps(status, indent=2, default=str)}")
    
    elif args.action == "list":
        endpoints = deployment_manager.list_active_endpoints()
        print(f"📋 Active Endpoints ({len(endpoints)}):")
        for endpoint in endpoints:
            print(f"  - {endpoint['name']}: {endpoint['status']}")
    
    elif args.action == "test" and args.endpoint_name and args.test_message:
        try:
            result = quick_inference_test(args.endpoint_name, args.test_message)
            print(f"🤖 Inference Result: {result}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    elif args.action == "delete" and args.endpoint_name:
        success = deployment_manager.delete_endpoint(args.endpoint_name)
        print(f"{'✅' if success else '❌'} Endpoint deletion {'successful' if success else 'failed'}")
    
    else:
        print("❌ Invalid arguments or missing required parameters")
        parser.print_help()