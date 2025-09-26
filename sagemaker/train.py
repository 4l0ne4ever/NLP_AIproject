#!/usr/bin/env python3
"""
SageMaker Training Script for Stranger Things Character Chatbot
"""
import subprocess
import sys

# Install required packages
print("Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets>=2.14.0", "accelerate>=0.20.0"])
print("Package installation complete.")

import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    
    # Model hyperparameters
    parser.add_argument("--base_model", type=str, default="microsoft/DialoGPT-medium")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=20)  # Reduced for CPU/demo
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--fp16", type=str, default="false")
    parser.add_argument("--huggingface_token", type=str, default="")

    # S3 upload configuration (passed via hyperparameters)
    parser.add_argument("--s3_bucket", type=str, default=os.environ.get("SM_S3_BUCKET", ""))
    parser.add_argument("--s3_region", type=str, default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument("--checkpoints_prefix", type=str, default="checkpoints/")
    parser.add_argument("--logs_prefix", type=str, default="logs/")
    parser.add_argument("--processed_prefix", type=str, default="data/processed/")
    parser.add_argument("--models_prefix", type=str, default="models/")
    parser.add_argument("--job_name", type=str, default=os.environ.get("TRAINING_JOB_NAME", ""))
    
    return parser.parse_args()

def load_training_data(data_path):
    """Load and prepare training data"""
    logger.info(f"Loading training data from {data_path}")
    
    # Look for CSV files in the data path
    csv_files = []
    if os.path.isdir(data_path):
        for file in os.listdir(data_path):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(data_path, file))
    else:
        csv_files = [data_path]
    
    logger.info(f"Found CSV files: {csv_files}")
    
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} rows from {csv_file}")
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")
    
    if not all_data:
        raise ValueError("No valid training data found!")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total training examples: {len(combined_df)}")
    
    return combined_df

def prepare_dataset(df, tokenizer, max_length=512):
    """Convert dataframe to HuggingFace dataset"""
    logger.info("Preparing dataset for training")
    
    # Create training texts by combining input and output
    texts = []
    for _, row in df.iterrows():
        if 'input' in df.columns and 'output' in df.columns:
            text = f"{row['input']}{tokenizer.eos_token}{row['output']}{tokenizer.eos_token}"
            texts.append(text)
    
    logger.info(f"Created {len(texts)} training texts")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors="pt"
        )
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    return tokenized_dataset

def _upload_directory_to_s3(local_dir, bucket, s3_prefix, region):
    """Recursively upload a local directory to S3 under s3_prefix"""
    import boto3, os
    s3 = boto3.client('s3', region_name=region)
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, start=local_dir)
            key = f"{s3_prefix.rstrip('/')}/{rel}".replace("\\", "/")
            s3.upload_file(local_path, bucket, key)


def main():
    """Main training function"""
    args = parse_args()
    
    logger.info("Starting SageMaker training")
    logger.info(f"Arguments: {args}")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    
    # Load and prepare data
    training_df = load_training_data(args.train)
    train_dataset = prepare_dataset(training_df, tokenizer)

    # Save a processed copy of the dataset (CSV) for auditability
    try:
        import tempfile
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_path = os.path.join(tmpdir, "training_data_processed.csv")
            # If training_df is already a DataFrame
            training_df.to_csv(processed_path, index=False)
            if args.s3_bucket:
                run_id = args.job_name or str(int(time.time()))
                s3_prefix = f"{args.processed_prefix.rstrip('/')}/{run_id}"
                _upload_directory_to_s3(tmpdir, args.s3_bucket, s3_prefix, args.s3_region)
                logger.info(f"Uploaded processed dataset to s3://{args.s3_bucket}/{s3_prefix}")
    except Exception as e:
        logger.warning(f"Failed to upload processed dataset: {e}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=args.max_steps,
        evaluation_strategy="no",
        save_strategy="steps",
        fp16=args.fp16.lower() == "true",
        logging_dir=f"{args.model_dir}/logs",
        report_to=None,  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {args.model_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.model_dir)
    
    # Save training info
    training_info = {
        "base_model": args.base_model,
        "training_examples": len(training_df),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate
    }
    
    with open(os.path.join(args.model_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    # Upload checkpoints and logs to S3 (in addition to SageMaker model artifacts)
    try:
        if args.s3_bucket:
            run_id = args.job_name or str(int(time.time()))
            # Upload checkpoints (checkpoint-* folders inside model_dir)
            ckpt_root = args.model_dir
            ckpt_prefix = f"{args.checkpoints_prefix.rstrip('/')}/{run_id}"
            # Only upload checkpoint directories/files
            import glob
            checkpoint_dirs = [p for p in glob.glob(os.path.join(ckpt_root, "checkpoint-*")) if os.path.isdir(p)]
            if checkpoint_dirs:
                for d in checkpoint_dirs:
                    _upload_directory_to_s3(d, args.s3_bucket, ckpt_prefix + '/' + os.path.basename(d), args.s3_region)
                logger.info(f"Uploaded checkpoints to s3://{args.s3_bucket}/{ckpt_prefix}")
            # Upload logs directory
            logs_dir = os.path.join(args.model_dir, "logs")
            if os.path.isdir(logs_dir):
                logs_prefix = f"{args.logs_prefix.rstrip('/')}/{run_id}"
                _upload_directory_to_s3(logs_dir, args.s3_bucket, logs_prefix, args.s3_region)
                logger.info(f"Uploaded logs to s3://{args.s3_bucket}/{logs_prefix}")
    except Exception as e:
        logger.warning(f"Failed to upload checkpoints/logs: {e}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()