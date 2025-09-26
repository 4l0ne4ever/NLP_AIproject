import torch
import gc
import os
import glob
import boto3
import json
from datetime import datetime
from pathlib import Path
import huggingface_hub
import pandas as pd
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import transformers

class S3CheckpointCallback(TrainerCallback):
    """Custom callback to upload checkpoints to S3 during training"""
    
    def __init__(self, s3_client, bucket_name, s3_prefix, upload_frequency=2):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.upload_frequency = upload_frequency  # Upload every N checkpoints
        self.checkpoint_count = 0
        
    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved"""
        self.checkpoint_count += 1
        
        # Only upload every Nth checkpoint to avoid excessive uploads
        if self.checkpoint_count % self.upload_frequency != 0:
            return
            
        try:
            output_dir = Path(args.output_dir)
            latest_checkpoint = max(
                [d for d in output_dir.glob("checkpoint-*") if d.is_dir()],
                key=lambda x: int(x.name.split("-")[1])
            )
            
            print(f"📤 Uploading checkpoint {latest_checkpoint.name} to S3...")
            
            for file_path in latest_checkpoint.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(output_dir)
                    s3_key = f"{self.s3_prefix}{rel_path.as_posix()}"
                    self.s3_client.upload_file(str(file_path), self.bucket_name, s3_key)
            
            print(f"✅ Uploaded {latest_checkpoint.name} to S3")
            
            # Also save training progress
            progress_info = {
                "checkpoint": latest_checkpoint.name,
                "step": state.global_step,
                "epoch": state.epoch,
                "timestamp": datetime.now().isoformat(),
                "training_loss": state.log_history[-1].get("train_loss") if state.log_history else None,
                "eval_loss": state.log_history[-1].get("eval_loss") if state.log_history else None
            }
            
            progress_key = f"{self.s3_prefix}training_progress.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=progress_key,
                Body=json.dumps(progress_info, indent=2),
                ContentType='application/json'
            )
            
        except Exception as e:
            print(f"⚠️ Failed to upload checkpoint to S3: {e}")
class CharacterChatbot():
    def __init__(self,
                model_path,
                data_path = "/content/data/transcripts/",
                huggingface_token=None,
                ):
        self.model_path = model_path
        data_path = os.path.abspath(data_path.rstrip('/')) + '/'
        if not os.path.isdir(data_path):
            raise ValueError(f"Data directory does not exist: {data_path}")
        self.data_files = glob.glob(os.path.join(data_path, "*.csv"))
        if not self.data_files:
            raise ValueError(f"No CSV files found in {data_path}. Directory exists: {os.path.exists(data_path)}")
        print(f"Successfully found {len(self.data_files)} CSV files in {data_path}:")
        self.huggingface_token = huggingface_token
        self.base_model_path = "meta-llama/Llama-3.2-3B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)
            
        # Only load model if the repo exists; do not auto-train in __init__
        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_model(self.model_path)
        else:
            print(f"Model {self.model_path} not found. Ready to train a new model when train() is called.")
            self.model = None
            
    def load_model(self, model_path):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
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
        return pipeline
            
    def load_data(self):
        transcript_df = pd.concat(
            [pd.read_csv(file, on_bad_lines="skip") for file in self.data_files],
            ignore_index=True
        )
        transcript_df = transcript_df.dropna()
        transcript_df['number_of_words'] = transcript_df['line'].str.strip().str.split(" ")
        transcript_df['number_of_words'] = transcript_df['number_of_words'].apply(lambda x: len(x))
        transcript_df['eleven_res'] = 0
        transcript_df.loc[(transcript_df['name']=="Eleven")&(transcript_df['number_of_words']>=1),'eleven_res']=1
        
        indexes_to_take = list(transcript_df[(transcript_df['eleven_res'] == 1)&(transcript_df.index>0)].index)
        system_prompt = """ You are Eleven (or El for short), a character from the Netflix series Stranger Things. Your responses should reflect her personality and speech patterns \n"""
        prompts = []
        for ind in indexes_to_take:
            prompt = system_prompt
            prompt += transcript_df.iloc[ind -1]['line']
            prompt += "\n"
            prompt += transcript_df.iloc[ind]['line']
            prompts.append(prompt)
            
        df = pd.DataFrame({"prompt": prompts})
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def train(self,
            base_model_name_or_path,
            dataset,
            output_dir="./results",
            per_device_train_batch_size=1,  # Keep safe
            gradient_accumulation_steps=1,  # Keep safe
            optimizer="paged_adamw_32bit",
            save_steps=200,
            logging_steps=10,
            learning_rate=5e-5,  # Stable LR
            max_grad_norm=0.3,
            max_steps=600,  # Moderate increase
            warmup_ratio=0.1,  # Reduced
            lr_scheduler_type="cosine",  # Smoother decay
            push_to_hub: bool = False,
            s3_bucket_name: str = None,
            run_id: str = None):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
        )
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        # Preprocess dataset with optimized settings
        def preprocess_function(examples):
            return tokenizer(
                examples["prompt"], 
                truncation=True, 
                padding="max_length", 
                max_length=256,  # Reduced for faster training
                return_tensors="pt"
            )

        dataset = dataset.map(preprocess_function, batched=True)

        # LoRA config
        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64  # Keep safe
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj"],  # Adjusted for Qwen
        )

        # Split dataset
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optimizer,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            fp16=True,  # Mixed precision for speed
            bf16=False,  # Use fp16 instead
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
            report_to="none",
            gradient_checkpointing=True,  # Save memory
            eval_strategy="steps",
            eval_steps=25,  # More frequent evaluation
            save_strategy="steps",
            save_total_limit=2,  # Keep fewer checkpoints for space
            load_best_model_at_end=False,  # Skip to save time
            logging_dir=f"{output_dir}/logs",
            dataloader_pin_memory=False,
            dataloader_num_workers=2,  # Parallel data loading
            remove_unused_columns=False,
            prediction_loss_only=True,  # Skip unnecessary computation
            disable_tqdm=False,  # Keep progress bars for visibility
        )

        # Initialize callbacks
        callbacks = []
        
        # Add S3 checkpoint callback if S3 details provided
        if s3_bucket_name and run_id:
            try:
                from config import S3_REGION
                s3_client = boto3.client('s3', region_name=S3_REGION)
                s3_prefix = f"live_checkpoints/{run_id}/"
                s3_callback = S3CheckpointCallback(
                    s3_client=s3_client,
                    bucket_name=s3_bucket_name,
                    s3_prefix=s3_prefix,
                    upload_frequency=5  # Upload every 5th checkpoint to reduce overhead
                )
                callbacks.append(s3_callback)
                print(f"📡 S3 live checkpoint saving enabled: s3://{s3_bucket_name}/{s3_prefix}")
            except Exception as e:
                print(f"Warning: Could not setup S3 live checkpointing: {e}")
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=training_args,
            callbacks=callbacks,
        )
        result = trainer.train()

        # Save and merge model
        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            return_dict=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, "final_ckpt")
        model = model.merge_and_unload()  # Merge LoRA weights

        # Always save merged model locally
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Optionally push to Hugging Face Hub
        if push_to_hub and self.huggingface_token is not None:
            try:
                model.push_to_hub(self.model_path)
                tokenizer.push_to_hub(self.model_path)
            except Exception as e:
                print(f"Warning: Failed to push to hub: {e}")

        # Flush memory
        del model, base_model, trainer
        gc.collect()
        torch.cuda.empty_cache()

        # Return metrics if available
        metrics = getattr(result, 'metrics', {}) if 'result' in locals() and result is not None else {}
        return metrics
        
    def chat(self, message, history):
        messages = []
        #TODO: add system prompt
        messages.append({"role":"system","content":"""You are Eleven (or El for short), a character from the Netflix series Stranger Things. Your responses should reflect her personality and speech patterns \n"""})
        
        for message_and_response in history:
            messages.append({
                "role": "user",
                "content": message_and_response[0]
            })
            messages.append({
                "role": "assistant",
                "content": message_and_response[1]
            })
            
        messages.append({
            "role": "user",
            "content": message
        })
        terminator= [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        output = self.model(
            messages,
            max_length=512,
            eos_token_id=terminator,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        output_message = output[0]['generated_text'][-1]
        return output_message