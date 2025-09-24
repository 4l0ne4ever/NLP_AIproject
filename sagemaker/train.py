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
    parser.add_argument("--max_steps", type=int, default=20)  # Reduced for CPU
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--fp16", type=str, default="false")
    parser.add_argument("--huggingface_token", type=str, default="")
    
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
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()