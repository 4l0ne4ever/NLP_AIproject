"""
SageMaker S3 Storage Manager for Stranger Things NLP Project

This module provides S3 storage management specifically optimized for SageMaker workflows,
including training data management, model artifact handling, and batch processing.
"""

import boto3
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm
import tarfile
import tempfile


class SageMakerS3Manager:
    """S3 storage manager optimized for SageMaker workflows"""
    
    def __init__(self, bucket_name: str, region: str = "us-east-1", aws_profile: Optional[str] = None):
        """
        Initialize SageMaker S3 storage manager
        
        Args:
            bucket_name: S3 bucket name for SageMaker artifacts
            region: AWS region
            aws_profile: AWS profile name (optional)
        """
        self.bucket_name = bucket_name
        self.region = region
        
        # Set up logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize S3 client
        try:
            if aws_profile:
                session = boto3.Session(profile_name=aws_profile)
                self.s3_client = session.client('s3', region_name=region)
            else:
                self.s3_client = boto3.client('s3', region_name=region)
                
            # Test connection and create bucket if needed
            self._ensure_bucket_exists()
            
            self.logger.info(f"Connected to S3 bucket: {bucket_name}")
            
        except NoCredentialsError:
            raise Exception(f"AWS credentials not found. Please configure AWS CLI or set environment variables.")
        except ClientError as e:
            raise Exception(f"Error connecting to S3: {e}")
    
    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists, create if it doesn't"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.logger.info(f"Creating S3 bucket: {self.bucket_name}")
                self._create_bucket()
            else:
                raise
    
    def _create_bucket(self):
        """Create S3 bucket with appropriate configuration"""
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            # Add bucket policies for SageMaker access
            self._setup_sagemaker_bucket_policy()
            
            self.logger.info(f"Created S3 bucket: {self.bucket_name}")
            
        except ClientError as e:
            self.logger.error(f"Error creating bucket: {e}")
            raise
    
    def _setup_sagemaker_bucket_policy(self):
        """Set up bucket policy for SageMaker access"""
        # This is a basic policy - you might want to customize based on your security requirements
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AllowSageMakerAccess",
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "sagemaker.amazonaws.com"
                    },
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{self.bucket_name}",
                        f"arn:aws:s3:::{self.bucket_name}/*"
                    ]
                }
            ]
        }
        
        try:
            self.s3_client.put_bucket_policy(
                Bucket=self.bucket_name,
                Policy=json.dumps(policy)
            )
            self.logger.info("Set up SageMaker bucket policy")
        except ClientError as e:
            self.logger.warning(f"Could not set bucket policy: {e}")
    
    def upload_file(self, local_path: Union[str, Path], s3_key: str, 
                   show_progress: bool = True) -> bool:
        """Upload a file to S3 with progress tracking"""
        local_path = Path(local_path)
        
        if not local_path.exists():
            self.logger.error(f"Local file does not exist: {local_path}")
            return False
        
        try:
            file_size = local_path.stat().st_size
            
            if show_progress and file_size > 10 * 1024 * 1024:  # Show progress for files > 10MB
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {local_path.name}") as pbar:
                    def callback(bytes_transferred):
                        pbar.update(bytes_transferred)
                    
                    self.s3_client.upload_file(
                        str(local_path), 
                        self.bucket_name, 
                        s3_key,
                        Callback=callback
                    )
            else:
                self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            
            self.logger.info(f"Uploaded {local_path.name} to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            self.logger.error(f"Error uploading {local_path}: {e}")
            return False
    
    def download_file(self, s3_key: str, local_path: Union[str, Path], 
                     show_progress: bool = True) -> bool:
        """Download a file from S3 with progress tracking"""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get file size for progress tracking
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            file_size = response['ContentLength']
            
            if show_progress and file_size > 10 * 1024 * 1024:  # Show progress for files > 10MB
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {s3_key}") as pbar:
                    def callback(bytes_transferred):
                        pbar.update(bytes_transferred)
                    
                    self.s3_client.download_file(
                        self.bucket_name, 
                        s3_key, 
                        str(local_path),
                        Callback=callback
                    )
            else:
                self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            
            self.logger.info(f"Downloaded {s3_key} to {local_path}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                self.logger.error(f"File not found in S3: {s3_key}")
            else:
                self.logger.error(f"Error downloading {s3_key}: {e}")
            return False
    
    def upload_directory(self, local_dir: Union[str, Path], s3_prefix: str, 
                        exclude_patterns: Optional[List[str]] = None) -> Dict[str, bool]:
        """Upload entire directory to S3 with SageMaker optimization"""
        local_dir = Path(local_dir)
        exclude_patterns = exclude_patterns or ['.DS_Store', '__pycache__', '*.pyc', '.git']
        results = {}
        
        if not local_dir.exists() or not local_dir.is_dir():
            self.logger.error(f"Directory does not exist: {local_dir}")
            return results
        
        # Get all files in directory
        files = []
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Check exclude patterns
                should_exclude = any(pattern in str(file_path) for pattern in exclude_patterns)
                if not should_exclude:
                    files.append(file_path)
        
        self.logger.info(f"Uploading {len(files)} files from {local_dir}")
        
        for file_path in tqdm(files, desc="Uploading files"):
            relative_path = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix.rstrip('/')}/{relative_path.as_posix()}"
            results[str(relative_path)] = self.upload_file(file_path, s3_key, show_progress=False)
        
        successful_uploads = sum(1 for success in results.values() if success)
        self.logger.info(f"Successfully uploaded {successful_uploads}/{len(files)} files")
        
        return results
    
    def download_directory(self, s3_prefix: str, local_dir: Union[str, Path]) -> Dict[str, bool]:
        """Download entire directory from S3"""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        try:
            # List all objects with prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)
            
            files = []
            for page in pages:
                if 'Contents' in page:
                    files.extend([obj['Key'] for obj in page['Contents']])
            
            self.logger.info(f"Downloading {len(files)} files to {local_dir}")
            
            for s3_key in tqdm(files, desc="Downloading files"):
                # Remove prefix to get relative path
                relative_path = s3_key[len(s3_prefix.rstrip('/')) + 1:]
                if not relative_path:  # Skip if it's just the prefix itself
                    continue
                    
                local_path = local_dir / relative_path
                results[relative_path] = self.download_file(s3_key, local_path, show_progress=False)
            
            successful_downloads = sum(1 for success in results.values() if success)
            self.logger.info(f"Successfully downloaded {successful_downloads}/{len(files)} files")
            
            return results
            
        except ClientError as e:
            self.logger.error(f"Error downloading directory: {e}")
            return results
    
    def create_model_package(self, model_dir: Union[str, Path], 
                           model_name: str, s3_prefix: str = "models/") -> str:
        """
        Create and upload a SageMaker-compatible model package
        
        Args:
            model_dir: Local directory containing model files
            model_name: Name for the model package
            s3_prefix: S3 prefix for model storage
        
        Returns:
            S3 URI of the model package
        """
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise ValueError(f"Model directory does not exist: {model_dir}")
        
        # Create model tar.gz package
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        self.logger.info(f"Creating model package for {model_name}")
        
        try:
            with tarfile.open(temp_path, 'w:gz') as tar:
                for file_path in model_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(model_dir)
                        tar.add(file_path, arcname=arcname)
            
            # Upload to S3
            s3_key = f"{s3_prefix.rstrip('/')}/{model_name}/model.tar.gz"
            success = self.upload_file(temp_path, s3_key)
            
            if success:
                s3_uri = f"s3://{self.bucket_name}/{s3_key}"
                self.logger.info(f"Model package uploaded: {s3_uri}")
                return s3_uri
            else:
                raise Exception("Failed to upload model package")
                
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    def download_model_package(self, s3_uri: str, extract_to: Union[str, Path]) -> bool:
        """
        Download and extract a SageMaker model package
        
        Args:
            s3_uri: S3 URI of the model package
            extract_to: Local directory to extract the model
        
        Returns:
            Success status
        """
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        
        # Parse S3 URI
        s3_key = s3_uri.replace(f"s3://{self.bucket_name}/", "")
        
        # Download model package
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            success = self.download_file(s3_key, temp_path)
            
            if success:
                # Extract tar.gz
                with tarfile.open(temp_path, 'r:gz') as tar:
                    tar.extractall(extract_to)
                
                self.logger.info(f"Model package extracted to: {extract_to}")
                return True
            else:
                return False
                
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    def list_objects(self, prefix: str = '', max_keys: int = 1000) -> List[Dict]:
        """List objects in S3 bucket with given prefix"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.bucket_name, 
                Prefix=prefix,
                PaginationConfig={'MaxItems': max_keys}
            )
            
            objects = []
            for page in pages:
                if 'Contents' in page:
                    objects.extend(page['Contents'])
            
            return objects
            
        except ClientError as e:
            self.logger.error(f"Error listing objects: {e}")
            return []
    
    def delete_object(self, s3_key: str) -> bool:
        """Delete an object from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            self.logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            self.logger.error(f"Error deleting {s3_key}: {e}")
            return False
    
    def save_json(self, data: Dict, s3_key: str) -> bool:
        """Save dictionary as JSON to S3"""
        try:
            json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_str.encode('utf-8'),
                ContentType='application/json'
            )
            self.logger.info(f"Saved JSON to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving JSON: {e}")
            return False
    
    def load_json(self, s3_key: str) -> Optional[Dict]:
        """Load JSON from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            json_str = response['Body'].read().decode('utf-8')
            return json.loads(json_str)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                self.logger.warning(f"JSON file not found: {s3_key}")
            else:
                self.logger.error(f"Error loading JSON: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}")
            return None
    
    def create_training_data_manifest(self, data_prefix: str) -> str:
        """
        Create a SageMaker training data manifest file
        
        Args:
            data_prefix: S3 prefix where training data is stored
        
        Returns:
            S3 URI of the manifest file
        """
        # List all data files
        data_objects = self.list_objects(data_prefix)
        data_files = [obj for obj in data_objects if not obj['Key'].endswith('/')]
        
        # Create manifest entries
        manifest_entries = []
        for obj in data_files:
            s3_uri = f"s3://{self.bucket_name}/{obj['Key']}"
            manifest_entries.append({
                "source-ref": s3_uri,
                "size": obj['Size'],
                "last-modified": obj['LastModified'].isoformat()
            })
        
        # Save manifest
        manifest_key = f"{data_prefix.rstrip('/')}/training-manifest.json"
        manifest_data = {
            "entries": manifest_entries,
            "total_files": len(manifest_entries),
            "created_at": "2024-01-01T00:00:00Z"  # Use current timestamp in real implementation
        }
        
        success = self.save_json(manifest_data, manifest_key)
        
        if success:
            return f"s3://{self.bucket_name}/{manifest_key}"
        else:
            raise Exception("Failed to create training data manifest")
    
    def get_model_artifacts(self, training_job_name: str) -> Optional[str]:
        """Get model artifacts S3 URI from a completed training job"""
        try:
            # This would typically query SageMaker API to get the model artifacts location
            # For now, we'll construct the expected path
            s3_uri = f"s3://{self.bucket_name}/models/{training_job_name}/output/model.tar.gz"
            
            # Check if the file exists
            s3_key = s3_uri.replace(f"s3://{self.bucket_name}/", "")
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                return s3_uri
            except ClientError:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting model artifacts: {e}")
            return None
    
    def cleanup_training_artifacts(self, training_job_name: str, keep_model: bool = True) -> bool:
        """
        Clean up training artifacts to save storage costs
        
        Args:
            training_job_name: Name of the training job
            keep_model: Whether to keep the final model artifacts
        
        Returns:
            Success status
        """
        try:
            cleanup_prefixes = [
                f"checkpoints/{training_job_name}/",
                f"logs/{training_job_name}/",
                f"data-capture/{training_job_name}/"
            ]
            
            if not keep_model:
                cleanup_prefixes.append(f"models/{training_job_name}/")
            
            total_deleted = 0
            for prefix in cleanup_prefixes:
                objects = self.list_objects(prefix)
                for obj in objects:
                    if self.delete_object(obj['Key']):
                        total_deleted += 1
            
            self.logger.info(f"Cleaned up {total_deleted} artifacts for job: {training_job_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up artifacts: {e}")
            return False


# Convenience functions for common SageMaker S3 operations
def create_sagemaker_bucket(bucket_name: str, region: str = "us-east-1") -> SageMakerS3Manager:
    """Create a SageMaker-optimized S3 bucket"""
    return SageMakerS3Manager(bucket_name=bucket_name, region=region)


def upload_training_data(local_data_path: str, bucket_name: str, 
                        model_type: str = "chatbot") -> str:
    """Quick upload training data to SageMaker S3 bucket"""
    s3_manager = SageMakerS3Manager(bucket_name)
    s3_prefix = f"data/training/{model_type}/"
    
    results = s3_manager.upload_directory(local_data_path, s3_prefix)
    
    if all(results.values()):
        return f"s3://{bucket_name}/{s3_prefix}"
    else:
        raise Exception("Failed to upload some training data files")


def download_model_artifacts(training_job_name: str, bucket_name: str, 
                           local_path: str) -> bool:
    """Quick download model artifacts from completed training job"""
    s3_manager = SageMakerS3Manager(bucket_name)
    
    model_s3_uri = s3_manager.get_model_artifacts(training_job_name)
    if model_s3_uri:
        return s3_manager.download_model_package(model_s3_uri, local_path)
    else:
        return False


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="SageMaker S3 Manager operations")
    parser.add_argument("--bucket-name", required=True, help="S3 bucket name")
    parser.add_argument("--operation", choices=["create", "upload", "download", "list"], required=True)
    parser.add_argument("--local-path", help="Local path for upload/download operations")
    parser.add_argument("--s3-prefix", help="S3 prefix for operations")
    
    args = parser.parse_args()
    
    # Create S3 manager
    s3_manager = SageMakerS3Manager(args.bucket_name)
    
    if args.operation == "create":
        print(f"S3 bucket '{args.bucket_name}' is ready for SageMaker operations")
    
    elif args.operation == "upload" and args.local_path and args.s3_prefix:
        local_path = Path(args.local_path)
        if local_path.is_dir():
            results = s3_manager.upload_directory(args.local_path, args.s3_prefix)
            print(f"Uploaded directory: {sum(results.values())}/{len(results)} files successful")
        else:
            success = s3_manager.upload_file(args.local_path, args.s3_prefix)
            print(f"Upload {'successful' if success else 'failed'}")
    
    elif args.operation == "download" and args.local_path and args.s3_prefix:
        success = s3_manager.download_directory(args.s3_prefix, args.local_path)
        print(f"Downloaded directory: {sum(success.values())}/{len(success)} files successful")
    
    elif args.operation == "list":
        objects = s3_manager.list_objects(args.s3_prefix or "")
        print(f"Found {len(objects)} objects:")
        for obj in objects[:10]:  # Show first 10
            print(f"  - {obj['Key']} ({obj['Size']} bytes)")
        if len(objects) > 10:
            print(f"  ... and {len(objects) - 10} more")
    
    else:
        print("Invalid operation or missing required arguments")
