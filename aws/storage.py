import boto3
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm

class S3StorageManager:
    """Comprehensive S3 storage manager for Stranger Things NLP project"""
    
    def __init__(self, bucket_name: str, aws_profile: Optional[str] = None):
        """
        Initialize S3 storage manager
        
        Args:
            bucket_name: S3 bucket name
            aws_profile: AWS profile name (optional)
        """
        self.bucket_name = bucket_name
        
        # Initialize S3 client
        try:
            if aws_profile:
                session = boto3.Session(profile_name=aws_profile)
                self.s3_client = session.client('s3')
            else:
                self.s3_client = boto3.client('s3')
                
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logging.info(f"Successfully connected to S3 bucket: {bucket_name}")
            
        except NoCredentialsError:
            raise Exception("AWS credentials not found. Please configure AWS CLI or set environment variables.")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.warning(f"Bucket {bucket_name} does not exist. Creating it...")
                self.create_bucket()
            else:
                raise Exception(f"Error connecting to S3: {e}")
    
    def create_bucket(self, region: str = 'us-east-1'):
        """Create S3 bucket if it doesn't exist"""
        try:
            if region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
            logging.info(f"Created S3 bucket: {self.bucket_name}")
        except ClientError as e:
            logging.error(f"Error creating bucket: {e}")
            raise
    
    def upload_file(self, local_path: Union[str, Path], s3_key: str, 
                   show_progress: bool = True) -> bool:
        """Upload a file to S3 with progress tracking"""
        local_path = Path(local_path)
        
        if not local_path.exists():
            logging.error(f"Local file does not exist: {local_path}")
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
            
            logging.info(f"Successfully uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logging.error(f"Error uploading {local_path}: {e}")
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
            
            logging.info(f"Successfully downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logging.error(f"File not found in S3: {s3_key}")
            else:
                logging.error(f"Error downloading {s3_key}: {e}")
            return False
    
    def upload_directory(self, local_dir: Union[str, Path], s3_prefix: str, 
                        exclude_patterns: Optional[List[str]] = None) -> Dict[str, bool]:
        """Upload entire directory to S3"""
        local_dir = Path(local_dir)
        exclude_patterns = exclude_patterns or []
        results = {}
        
        if not local_dir.exists() or not local_dir.is_dir():
            logging.error(f"Directory does not exist: {local_dir}")
            return results
        
        # Get all files in directory
        files = []
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Check exclude patterns
                should_exclude = any(pattern in str(file_path) for pattern in exclude_patterns)
                if not should_exclude:
                    files.append(file_path)
        
        logging.info(f"Uploading {len(files)} files from {local_dir}")
        
        for file_path in tqdm(files, desc="Uploading files"):
            relative_path = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix.rstrip('/')}/{relative_path.as_posix()}"
            results[str(relative_path)] = self.upload_file(file_path, s3_key, show_progress=False)
        
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
            
            logging.info(f"Downloading {len(files)} files to {local_dir}")
            
            for s3_key in tqdm(files, desc="Downloading files"):
                # Remove prefix to get relative path
                relative_path = s3_key[len(s3_prefix.rstrip('/')) + 1:]
                local_path = local_dir / relative_path
                results[relative_path] = self.download_file(s3_key, local_path, show_progress=False)
            
            return results
            
        except ClientError as e:
            logging.error(f"Error downloading directory: {e}")
            return results
    
    def list_objects(self, prefix: str = '') -> List[Dict]:
        """List objects in S3 bucket with given prefix"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            objects = []
            for page in pages:
                if 'Contents' in page:
                    objects.extend(page['Contents'])
            
            return objects
            
        except ClientError as e:
            logging.error(f"Error listing objects: {e}")
            return []
    
    def delete_object(self, s3_key: str) -> bool:
        """Delete an object from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logging.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logging.error(f"Error deleting {s3_key}: {e}")
            return False
    
    def save_json(self, data: Dict, s3_key: str) -> bool:
        """Save dictionary as JSON to S3"""
        try:
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_str.encode('utf-8'),
                ContentType='application/json'
            )
            logging.info(f"Saved JSON to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logging.error(f"Error saving JSON: {e}")
            return False
    
    def load_json(self, s3_key: str) -> Optional[Dict]:
        """Load JSON from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            json_str = response['Body'].read().decode('utf-8')
            return json.loads(json_str)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logging.warning(f"JSON file not found: {s3_key}")
            else:
                logging.error(f"Error loading JSON: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON: {e}")
            return None


# Convenience functions for the project
class StrangerThingsS3Manager(S3StorageManager):
    """Specialized S3 manager for Stranger Things project structure"""
    
    def __init__(self, bucket_name: str = 'stranger-things-nlp', aws_profile: Optional[str] = None):
        super().__init__(bucket_name, aws_profile)
        
        # Project-specific S3 structure
        self.paths = {
            'training_data': 'data/training/',
            'transcripts': 'data/transcripts/',
            'subtitle_data': 'data/subtitles/',
            'models': 'models/',
            'results': 'results/',
            'evaluations': 'evaluations/',
            'checkpoints': 'checkpoints/',
            'logs': 'logs/'
        }
    
    def upload_training_data(self, local_data_dir: Union[str, Path]) -> Dict[str, bool]:
        """Upload training data directory"""
        return self.upload_directory(
            local_data_dir, 
            self.paths['training_data'],
            exclude_patterns=['.DS_Store', '__pycache__', '*.pyc']
        )
    
    def download_training_data(self, local_data_dir: Union[str, Path]) -> Dict[str, bool]:
        """Download training data directory"""
        return self.download_directory(self.paths['training_data'], local_data_dir)
    
    def upload_model(self, model_name: str, local_model_dir: Union[str, Path]) -> Dict[str, bool]:
        """Upload trained model"""
        s3_prefix = f"{self.paths['models']}{model_name}/"
        return self.upload_directory(local_model_dir, s3_prefix)
    
    def download_model(self, model_name: str, local_model_dir: Union[str, Path]) -> Dict[str, bool]:
        """Download trained model"""
        s3_prefix = f"{self.paths['models']}{model_name}/"
        return self.download_directory(s3_prefix, local_model_dir)
    
    def save_training_results(self, model_name: str, results: Dict) -> bool:
        """Save training results as JSON"""
        s3_key = f"{self.paths['results']}{model_name}_results.json"
        return self.save_json(results, s3_key)
    
    def load_training_results(self, model_name: str) -> Optional[Dict]:
        """Load training results"""
        s3_key = f"{self.paths['results']}{model_name}_results.json"
        return self.load_json(s3_key)
    
    def save_evaluation_metrics(self, model_name: str, metrics: Dict) -> bool:
        """Save evaluation metrics"""
        s3_key = f"{self.paths['evaluations']}{model_name}_evaluation.json"
        return self.save_json(metrics, s3_key)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models in S3"""
        objects = self.list_objects(self.paths['models'])
        models = set()
        for obj in objects:
            # Extract model name from path like 'models/chatbot_v1/config.json'
            path_parts = obj['Key'].split('/')
            if len(path_parts) >= 3:  # models/model_name/file
                models.add(path_parts[1])
        return list(models)
