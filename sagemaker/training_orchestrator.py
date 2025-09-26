"""
SageMaker Training Orchestrator for Stranger Things NLP Project

This module handles SageMaker training job management including:
- Character chatbot training (Llama/Qwen)
- Text classification training
- Model packaging and deployment preparation
"""

import boto3
import json
import time
import logging
import os
from typing import Dict, List, Optional, Union
from pathlib import Path
import tarfile
import tempfile
from botocore.exceptions import ClientError

from config import SageMakerConfigManager
from storage import SageMakerS3Manager


class SageMakerTrainingOrchestrator:
    """Orchestrate SageMaker training jobs for different model types"""
    
    def __init__(self, config: SageMakerConfigManager = None):
        self.config = config or SageMakerConfigManager()
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.config.s3_config.region)
        self.s3_manager = SageMakerS3Manager(
            bucket_name=self.config.s3_config.bucket_name,
            region=self.config.s3_config.region
        )
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Track active training jobs
        self.active_jobs = {}
    
    def prepare_training_data(self, local_data_path: str, model_type: str = "chatbot") -> str:
        """
        Prepare and upload training data to S3
        
        Args:
            local_data_path: Path to local training data
            model_type: Type of model (chatbot, text_classifier, theme_classifier)
        
        Returns:
            S3 URI of uploaded training data
        """
        self.logger.info(f"Preparing training data for {model_type}")
        
        # Create training data package
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy and structure data based on model type
            if model_type == "chatbot":
                self._prepare_chatbot_data(local_data_path, temp_path)
            elif model_type == "text_classifier":
                self._prepare_text_classifier_data(local_data_path, temp_path)
            elif model_type == "theme_classifier":
                self._prepare_theme_classifier_data(local_data_path, temp_path)
            
            # Upload to S3
            s3_key = f"{self.config.s3_config.training_data_path}{model_type}/{int(time.time())}"
            upload_results = self.s3_manager.upload_directory(
                temp_path, 
                s3_key,
                exclude_patterns=['.DS_Store', '__pycache__', '*.pyc']
            )
            
            s3_uri = self.config.get_s3_uri('training_data', f"{model_type}/{int(time.time())}")
            self.logger.info(f"Training data uploaded to: {s3_uri}")
            
            return s3_uri
    
    def _prepare_chatbot_data(self, local_data_path: str, temp_path: Path):
        """Prepare chatbot training data"""
        import pandas as pd
        import shutil
        
        # Copy CSV files
        local_path = Path(local_data_path)
        if local_path.is_file() and local_path.suffix == '.csv':
            shutil.copy2(local_path, temp_path / 'training_data.csv')
        elif local_path.is_dir():
            csv_files = list(local_path.glob('**/*.csv'))
            if csv_files:
                # Combine all CSV files with proper quoting
                dfs = []
                for f in csv_files:
                    try:
                        df = pd.read_csv(f, quotechar='"', escapechar='\\')
                        dfs.append(df)
                    except Exception as e:
                        self.logger.warning(f"Could not read {f}: {e}")
                
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    combined_df.to_csv(temp_path / 'training_data.csv', index=False, quotechar='"')
        
        # Create metadata file
        metadata = {
            'model_type': 'character_chatbot',
            'data_format': 'csv',
            'columns': ['line', 'name'] if Path(local_data_path).is_dir() else None,
            'character': 'Eleven',
            'preprocessing_required': True
        }
        
        with open(temp_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _prepare_text_classifier_data(self, local_data_path: str, temp_path: Path):
        """Prepare text classification training data"""
        import shutil
        
        local_path = Path(local_data_path)
        
        # Copy JSONL or CSV files
        if local_path.is_file():
            shutil.copy2(local_path, temp_path / f'training_data{local_path.suffix}')
        elif local_path.is_dir():
            # Copy all relevant files
            for file_path in local_path.glob('**/*'):
                if file_path.is_file() and file_path.suffix in ['.jsonl', '.csv', '.json']:
                    shutil.copy2(file_path, temp_path / file_path.name)
        
        # Create metadata
        metadata = {
            'model_type': 'text_classifier',
            'task': 'location_classification',
            'preprocessing_required': True
        }
        
        with open(temp_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _prepare_theme_classifier_data(self, local_data_path: str, temp_path: Path):
        """Prepare theme classification data"""
        import shutil
        
        # Theme classifier uses subtitle files
        local_path = Path(local_data_path)
        
        if local_path.is_dir():
            # Copy subtitle files
            for file_path in local_path.glob('**/*.srt'):
                rel_path = file_path.relative_to(local_path)
                target_path = temp_path / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, target_path)
        
        metadata = {
            'model_type': 'theme_classifier',
            'data_format': 'srt',
            'preprocessing_required': True
        }
        
        with open(temp_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_training_script(self, model_type: str, output_path: str = None) -> str:
        """
        Create and upload training script for SageMaker
        
        Args:
            model_type: Type of model to train
            output_path: Local path to save the script (optional)
        
        Returns:
            S3 URI of uploaded training script
        """
        script_content = self._get_training_script_content(model_type)
        
        # Save script locally if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(script_content)
        
        # Upload to S3
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        s3_key = f"code/{model_type}_training_script.py"
        success = self.s3_manager.upload_file(script_path, s3_key)
        
        if success:
            s3_uri = f"s3://{self.config.s3_config.bucket_name}/{s3_key}"
            self.logger.info(f"Training script uploaded to: {s3_uri}")
            return s3_uri
        else:
            raise Exception("Failed to upload training script")
    
    def _get_training_script_content(self, model_type: str) -> str:
        """Generate training script content based on model type"""
        
        if model_type == "chatbot":
            return '''#!/usr/bin/env python3
"""
SageMaker Training Script for Character Chatbot
"""

import argparse
import json
import logging
import os
import sys
import pandas as pd
import torch
from pathlib import Path

# Import required libraries
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset
import huggingface_hub

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    
    # Model hyperparameters
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--fp16", type=str, default="true")
    parser.add_argument("--gradient_checkpointing", type=str, default="true")
    
    # HuggingFace settings
    parser.add_argument("--huggingface_token", type=str, default="")
    parser.add_argument("--model_name", type=str, default="stranger-things-chatbot")
    
    return parser.parse_args()

def load_and_process_data(train_dir):
    """Load and process training data"""
    logger.info(f"Loading data from {train_dir}")
    
    # Find CSV file
    csv_files = list(Path(train_dir).glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {train_dir}")
    
    # Load data
    df = pd.read_csv(csv_files[0])
    df = df.dropna()
    
    # Process for character-specific responses (Eleven)
    df['number_of_words'] = df['line'].str.strip().str.split().str.len()
    df['eleven_res'] = 0
    df.loc[(df['name'] == "Eleven") & (df['number_of_words'] >= 1), 'eleven_res'] = 1
    
    # Create conversation pairs
    indexes_to_take = list(df[(df['eleven_res'] == 1) & (df.index > 0)].index)
    
    system_prompt = "You are Eleven (or El for short), a character from the Netflix series Stranger Things. Your responses should reflect her personality and speech patterns.\\n"
    
    prompts = []
    for ind in indexes_to_take:
        prompt = system_prompt
        prompt += df.iloc[ind - 1]['line'] + "\\n"
        prompt += df.iloc[ind]['line']
        prompts.append(prompt)
    
    # Create dataset
    training_df = pd.DataFrame({"prompt": prompts})
    dataset = Dataset.from_pandas(training_df)
    
    logger.info(f"Created dataset with {len(dataset)} examples")
    return dataset

def train_model(args, dataset):
    """Train the character chatbot model"""
    logger.info("Starting model training")
    
    # Set up quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Preprocess dataset
    def preprocess_function(examples):
        return tokenizer(
            examples["prompt"], 
            truncation=True, 
            padding="max_length", 
            max_length=512
        )
    
    dataset = dataset.map(preprocess_function, batched=True)
    
    # LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"],
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/results",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=200,
        logging_steps=10,
        learning_rate=args.learning_rate,
        fp16=args.fp16.lower() == "true",
        max_grad_norm=0.3,
        max_steps=args.max_steps,
        warmup_ratio=0.1,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing.lower() == "true",
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_args,
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    logger.info("Training completed successfully")
    
    # Save training metrics
    if trainer.state.log_history:
        metrics = {
            "final_train_loss": trainer.state.log_history[-1].get('train_loss', 'N/A'),
            "final_eval_loss": trainer.state.log_history[-1].get('eval_loss', 'N/A'),
            "total_steps": trainer.state.global_step
        }
        
        with open(f"{args.output_data_dir}/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    args = parse_args()
    
    # Login to HuggingFace if token provided
    if args.huggingface_token:
        huggingface_hub.login(args.huggingface_token)
    
    # Load and process data
    dataset = load_and_process_data(args.train)
    
    # Train model
    train_model(args, dataset)
    
    logger.info("Training job completed successfully")
'''
        
        elif model_type == "text_classifier":
            return '''#!/usr/bin/env python3
"""
SageMaker Training Script for Text Classification
"""

import argparse
import json
import logging
import os
from pathlib import Path
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker arguments
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    
    return parser.parse_args()

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def load_and_process_data(train_dir):
    """Load and process training data"""
    logger.info(f"Loading data from {train_dir}")
    
    # Find data files
    data_files = list(Path(train_dir).glob("*.jsonl")) + list(Path(train_dir).glob("*.csv"))
    
    if not data_files:
        raise ValueError(f"No data files found in {train_dir}")
    
    # Load data based on file type
    if data_files[0].suffix == ".jsonl":
        df = pd.read_json(data_files[0], lines=True)
    else:
        df = pd.read_csv(data_files[0])
    
    # Assume columns are 'text' and 'label'
    if 'text' not in df.columns or 'label' not in df.columns:
        logger.warning("Expected columns 'text' and 'label' not found. Using first two columns.")
        df.columns = ['text', 'label'] + list(df.columns[2:])
    
    return df

def train_model(args, df):
    """Train the text classification model"""
    logger.info("Starting text classification training")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['encoded_labels'] = label_encoder.fit_transform(df['label'])
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['encoded_labels'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['encoded_labels']
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_encoder.classes_)
    )
    
    # Tokenize data
    def tokenize_function(texts):
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )
    
    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    
    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/results",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="/tmp/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=args.learning_rate,
    )
    
    # Create trainer
    trainer = CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train model
    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    # Save label encoder
    import joblib
    joblib.dump(label_encoder, f"{args.model_dir}/label_encoder.joblib")
    
    # Save training metrics
    if trainer.state.log_history:
        metrics = {
            "final_train_loss": trainer.state.log_history[-1].get('train_loss', 'N/A'),
            "final_eval_loss": trainer.state.log_history[-1].get('eval_loss', 'N/A'),
            "num_classes": len(label_encoder.classes_),
            "class_names": label_encoder.classes_.tolist()
        }
        
        with open(f"{args.output_data_dir}/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    args = parse_args()
    
    # Load and process data
    df = load_and_process_data(args.train)
    
    # Train model
    train_model(args, df)
    
    logger.info("Training job completed successfully")
'''
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def launch_training_job(self, 
                          job_name: str,
                          model_type: str,
                          training_data_s3_uri: str,
                          hyperparameters: Dict = None) -> str:
        """
        Launch a SageMaker training job
        
        Args:
            job_name: Unique name for the training job
            model_type: Type of model to train
            training_data_s3_uri: S3 URI of training data
            hyperparameters: Optional hyperparameters override
        
        Returns:
            Training job ARN
        """
        
        # Prepare training script
        script_s3_uri = self.create_training_script(model_type)
        
        # Merge hyperparameters
        final_hyperparameters = self.config.training_config.hyperparameters.copy()
        if hyperparameters:
            final_hyperparameters.update(hyperparameters)
        
        # Add HuggingFace token if available
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            final_hyperparameters['huggingface_token'] = hf_token
        
        # Pass S3 and job metadata to the training script so it can upload logs/ckpts/processed data
        final_hyperparameters['s3_bucket'] = self.config.s3_config.bucket_name
        final_hyperparameters['s3_region'] = self.config.s3_config.region
        final_hyperparameters['checkpoints_prefix'] = self.config.s3_config.checkpoints_path
        final_hyperparameters['logs_prefix'] = getattr(self.config.s3_config, 'logs_path', 'logs/')
        final_hyperparameters['processed_prefix'] = getattr(self.config.s3_config, 'processed_data_path', 'data/processed/')
        final_hyperparameters['models_prefix'] = self.config.s3_config.model_artifacts_path
        final_hyperparameters['job_name'] = job_name
        
        # Training job definition
        training_job_def = {
            'TrainingJobName': job_name,
            'RoleArn': self.config.training_config.role_arn,
            'AlgorithmSpecification': {
                'TrainingImage': self.config.training_config.training_image,
                'TrainingInputMode': self.config.training_config.input_mode,
                'EnableSageMakerMetricsTimeSeries': True
            },
            'InputDataConfig': [
                {
                    'ChannelName': 'train',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': training_data_s3_uri,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
'ContentType': 'text/csv',
                    'InputMode': self.config.training_config.input_mode
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': self.config.training_config.output_path
            },
            'ResourceConfig': {
                'InstanceType': self.config.training_config.instance_type,
                'InstanceCount': self.config.training_config.instance_count,
                'VolumeSizeInGB': self.config.training_config.volume_size_gb
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': self.config.training_config.max_runtime_seconds
            },
            'HyperParameters': {k: str(v) for k, v in final_hyperparameters.items()},
            'Tags': [
                {'Key': 'Project', 'Value': 'stranger-things-nlp'},
                {'Key': 'ModelType', 'Value': model_type},
                {'Key': 'Environment', 'Value': 'sagemaker'}
            ]
        }
        
        # Add checkpointing if enabled
        if self.config.training_config.enable_checkpointing:
            training_job_def['CheckpointConfig'] = {
                'S3Uri': self.config.training_config.checkpoint_s3_uri,
                'LocalPath': '/opt/ml/checkpoints'
            }
        
        try:
            # Create training job
            response = self.sagemaker_client.create_training_job(**training_job_def)
            
            # Track the job
            self.active_jobs[job_name] = {
                'arn': response['TrainingJobArn'],
                'model_type': model_type,
                'status': 'InProgress',
                'start_time': time.time()
            }
            
            self.logger.info(f"Training job '{job_name}' launched successfully!")
            self.logger.info(f"Job ARN: {response['TrainingJobArn']}")
            self.logger.info(f"Monitor in AWS Console: https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")
            
            return response['TrainingJobArn']
            
        except ClientError as e:
            self.logger.error(f"Failed to launch training job: {e}")
            raise
    
    def get_job_status(self, job_name: str) -> Dict:
        """Get status of a training job"""
        try:
            response = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
            
            status_info = {
                'job_name': job_name,
                'status': response['TrainingJobStatus'],
                'creation_time': response['CreationTime'],
                'training_start_time': response.get('TrainingStartTime'),
                'training_end_time': response.get('TrainingEndTime'),
                'model_artifacts': response.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
                'failure_reason': response.get('FailureReason')
            }
            
            # Update local tracking
            if job_name in self.active_jobs:
                self.active_jobs[job_name]['status'] = response['TrainingJobStatus']
            
            return status_info
            
        except ClientError as e:
            self.logger.error(f"Error getting job status: {e}")
            return {'job_name': job_name, 'status': 'Unknown', 'error': str(e)}
    
    def list_training_jobs(self, max_results: int = 10) -> List[Dict]:
        """List recent training jobs"""
        try:
            response = self.sagemaker_client.list_training_jobs(
                MaxResults=max_results,
                SortBy='CreationTime',
                SortOrder='Descending'
            )
            
            return [
                {
                    'job_name': job['TrainingJobName'],
                    'status': job['TrainingJobStatus'],
                    'creation_time': job['CreationTime'],
                    'training_end_time': job.get('TrainingEndTime')
                }
                for job in response['TrainingJobSummary']
                if job['TrainingJobName'].startswith('stranger-things')
            ]
            
        except ClientError as e:
            self.logger.error(f"Error listing training jobs: {e}")
            return []
    
    def stop_training_job(self, job_name: str) -> bool:
        """Stop a running training job"""
        try:
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)
            self.logger.info(f"Stopped training job: {job_name}")
            
            if job_name in self.active_jobs:
                self.active_jobs[job_name]['status'] = 'Stopping'
            
            return True
            
        except ClientError as e:
            self.logger.error(f"Error stopping training job: {e}")
            return False


# Convenience function for quick training job launch
def launch_chatbot_training(character: str = "eleven",
                          model_type: str = "llama",
                          local_data_path: str = None,
                          config: SageMakerConfigManager = None) -> str:
    """Quick launch for chatbot training"""
    
    orchestrator = SageMakerTrainingOrchestrator(config)
    
    # Prepare training data
    if local_data_path:
        training_data_uri = orchestrator.prepare_training_data(local_data_path, "chatbot")
    else:
        # Use default S3 location
        training_data_uri = orchestrator.config.get_s3_uri('training_data', 'chatbot')
    
    # Generate unique job name
    job_name = f"stranger-things-{character}-{model_type}-{int(time.time())}"
    
    # Set model-specific hyperparameters
    hyperparameters = {}
    if model_type.lower() == "llama":
        hyperparameters['base_model'] = 'meta-llama/Llama-3.2-3B-Instruct'
    elif model_type.lower() == "qwen":
        hyperparameters['base_model'] = 'Qwen/Qwen2.5-3B-Instruct'
    
    # Launch training job
    return orchestrator.launch_training_job(
        job_name=job_name,
        model_type="chatbot", 
        training_data_s3_uri=training_data_uri,
        hyperparameters=hyperparameters
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch SageMaker training jobs")
    parser.add_argument("--model-type", choices=["chatbot", "text_classifier", "theme_classifier"], required=True)
    parser.add_argument("--data-path", required=True, help="Local path to training data")
    parser.add_argument("--job-name", help="Custom job name")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = SageMakerTrainingOrchestrator()
    
    # Prepare training data
    training_data_uri = orchestrator.prepare_training_data(args.data_path, args.model_type)
    
    # Generate job name if not provided
    if not args.job_name:
        args.job_name = f"stranger-things-{args.model_type}-{int(time.time())}"
    
    # Launch training job
    job_arn = orchestrator.launch_training_job(
        job_name=args.job_name,
        model_type=args.model_type,
        training_data_s3_uri=training_data_uri
    )
    
    print(f"✅ Training job launched: {job_arn}")
    print(f"🔍 Track progress with: python -c \"from sagemaker.training_orchestrator import SageMakerTrainingOrchestrator; o = SageMakerTrainingOrchestrator(); print(o.get_job_status('{args.job_name}'))\"")