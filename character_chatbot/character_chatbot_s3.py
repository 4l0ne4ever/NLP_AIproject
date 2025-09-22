"""
S3-Enhanced Character Chatbot for AWS Deployment

This is an example of how to modify the existing character_chatbot.py to work with S3 storage.
The main changes are:
1. Download training data from S3 instead of local paths
2. Upload trained models to S3
3. Save training results and metrics to S3
4. Use AWS-optimized paths and configurations
"""

import torch
import gc
import os
import tempfile
import huggingface_hub
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import transformers

# Import our AWS modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from aws.storage import StrangerThingsS3Manager
from aws.config import AWSConfigManager

class S3CharacterChatbot:
    """S3-enhanced Character Chatbot for AWS deployment"""
    
    def __init__(self,
                 model_path,
                 s3_data_path="data/transcripts/",
                 huggingface_token=None,
                 aws_config=None):
        
        self.model_path = model_path
        self.s3_data_path = s3_data_path
        self.huggingface_token = huggingface_token
        
        # Initialize AWS components
        self.aws_config = aws_config or AWSConfigManager()
        self.s3_manager = StrangerThingsS3Manager(
            bucket_name=self.aws_config.s3_config.bucket_name
        )
        
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Login to HuggingFace if token provided
        if self.huggingface_token:
            huggingface_hub.login(self.huggingface_token)
        
        # Check if model exists, if not train it
        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_model(self.model_path)
        else:
            print(f"Model {self.model_path} not found. Starting training...")
            train_dataset = self.load_data_from_s3()
            self.train(train_dataset)
            self.model = self.load_model(self.model_path)
    
    def load_data_from_s3(self):
        """Download and load training data from S3"""
        print("Downloading training data from S3...")
        
        # Create temporary directory for data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_data_path = Path(temp_dir) / "training_data"
            temp_data_path.mkdir(exist_ok=True)
            
            # Download training data from S3
            results = self.s3_manager.download_directory(
                self.s3_data_path, 
                temp_data_path
            )
            
            print(f"Downloaded {len(results)} files from S3")
            
            # Find CSV files in the downloaded data
            csv_files = list(temp_data_path.glob("**/*.csv"))
            
            if not csv_files:
                raise ValueError(f"No CSV files found in downloaded S3 data: {temp_data_path}")
            
            print(f"Found {len(csv_files)} CSV files for training")
            
            # Load and process the data
            transcript_df = pd.concat(
                [pd.read_csv(file, on_bad_lines="skip") for file in csv_files],
                ignore_index=True
            )
            
        return self._process_training_data(transcript_df)
    
    def _process_training_data(self, transcript_df):
        """Process training data for character-specific responses"""
        print("Processing training data...")
        
        transcript_df = transcript_df.dropna()
        transcript_df['number_of_words'] = transcript_df['line'].str.strip().str.split(" ")
        transcript_df['number_of_words'] = transcript_df['number_of_words'].apply(lambda x: len(x))
        transcript_df['eleven_res'] = 0
        transcript_df.loc[
            (transcript_df['name'] == "Eleven") & (transcript_df['number_of_words'] >= 1), 
            'eleven_res'
        ] = 1
        
        indexes_to_take = list(transcript_df[
            (transcript_df['eleven_res'] == 1) & (transcript_df.index > 0)
        ].index)
        
        system_prompt = """You are Eleven (or El for short), a character from the Netflix series Stranger Things. Your responses should reflect her personality and speech patterns \n"""
        
        prompts = []
        for ind in indexes_to_take:
            prompt = system_prompt
            prompt += transcript_df.iloc[ind - 1]['line']
            prompt += "\n"
            prompt += transcript_df.iloc[ind]['line']
            prompts.append(prompt)
        
        df = pd.DataFrame({"prompt": prompts})
        dataset = Dataset.from_pandas(df)
        
        print(f"Created training dataset with {len(dataset)} examples")
        return dataset
    
    def train(self, dataset):
        """Train the model with S3 integration"""
        print("Starting model training...")
        
        # Training configuration from AWS config
        training_config = self.aws_config.training_config
        
        # Set up model and tokenizer
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model_path = training_config.base_model_llama
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Preprocess dataset
        def preprocess_function(examples):
            return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)
        
        dataset = dataset.map(preprocess_function, batched=True)
        
        # LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            r=training_config.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        # Training arguments with AWS optimization
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            optim="paged_adamw_32bit",
            save_steps=200,
            logging_steps=10,
            learning_rate=training_config.learning_rate,
            fp16=training_config.fp16,
            max_grad_norm=0.3,
            max_steps=training_config.max_steps,
            warmup_ratio=0.1,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="none",
            gradient_checkpointing=training_config.gradient_checkpointing,
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
        
        print("Training started...")
        trainer.train()
        
        # Save model locally first
        local_model_path = "final_ckpt"
        trainer.model.save_pretrained(local_model_path)
        tokenizer.save_pretrained(local_model_path)
        
        # Merge and upload model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            return_dict=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        
        model = PeftModel.from_pretrained(base_model, local_model_path)
        model = model.merge_and_unload()
        
        # Push to HuggingFace Hub
        print("Uploading model to HuggingFace Hub...")
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)
        
        # Upload model to S3 as backup
        print("Backing up model to S3...")
        model_name = self.model_path.split('/')[-1]
        self.s3_manager.upload_model(model_name, local_model_path)
        
        # Save training results to S3
        training_results = {
            "model_path": self.model_path,
            "training_steps": training_args.max_steps,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "final_train_loss": trainer.state.log_history[-1].get('train_loss', 'N/A'),
            "final_eval_loss": trainer.state.log_history[-1].get('eval_loss', 'N/A'),
            "device": self.device,
            "training_completed": True
        }
        
        self.s3_manager.save_training_results(model_name, training_results)
        print("Training results saved to S3")
        
        # Clean up memory
        del model, base_model, trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        print("Training completed successfully!")
    
    def load_model(self, model_path):
        """Load model for inference"""
        print(f"Loading model: {model_path}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": bnb_config,
            },
        )
        
        print("Model loaded successfully")
        return pipeline
    
    def chat(self, message, history):
        """Chat with the character using the trained model"""
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system",
            "content": """You are Eleven (or El for short), a character from the Netflix series Stranger Things. Your responses should reflect her personality and speech patterns \n"""
        })
        
        # Add conversation history
        for message_and_response in history:
            messages.append({
                "role": "user",
                "content": message_and_response[0]
            })
            messages.append({
                "role": "assistant",
                "content": message_and_response[1]
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Format prompt
        prompt = self.model.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Generate response
        response = self.model(
            prompt,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.model.tokenizer.eos_token_id
        )
        
        # Extract the generated text
        generated_text = response[0]['generated_text']
        assistant_response = generated_text.split("assistant\n")[-1].strip()
        
        return assistant_response
    
    def get_model_info(self):
        """Get information about the current model"""
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "s3_bucket": self.s3_manager.bucket_name,
            "s3_data_path": self.s3_data_path,
        }
        
        # Try to get training results from S3
        model_name = self.model_path.split('/')[-1]
        training_results = self.s3_manager.load_training_results(model_name)
        if training_results:
            info.update(training_results)
        
        return info

# Example usage for AWS deployment
if __name__ == "__main__":
    # This would be called by the EC2 training script
    import os
    
    # Get configuration
    aws_config = AWSConfigManager()
    
    # Create S3-enabled chatbot
    chatbot = S3CharacterChatbot(
        model_path="christopherxzyx/StrangerThings_Llama-3-4B_ec2_s3",
        s3_data_path="data/transcripts/",
        huggingface_token=os.getenv('HUGGINGFACE_TOKEN'),
        aws_config=aws_config
    )
    
    # Test the chatbot
    print("Testing the chatbot...")
    response = chatbot.chat("Hello Eleven, how are you?", [])
    print(f"Eleven: {response}")
    
    # Print model info
    info = chatbot.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")