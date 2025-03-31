import torch
import gc
import huggingface_hub
import pandas as pd
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import transformers
class CharacterChatbot():
    def __init__(self,
                model_path,
                data_path = ["/content/data/transcripts/transcriptS01E08.csv",
                            "/content/data/transcripts/transcriptS02E09.csv",
                            "/content/data/transcripts/transcriptS03E08.csv",
                            "/content/data/transcripts/transcriptS04E09.csv",],
                huggingface_token=None,
                ):
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = "meta-llama/Llama-3.2-3B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)
            
        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_model(self.model_path)
        else:
            print(f"Model {self.model_path} not found. We will train our model.")
            train_dataset = self.load_data()
            #TODO: train
            self.train(self.base_model_path,train_dataset)
            #TODO: load model
            self.model = self.load_model(self.model_path)
            
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
            [pd.read_csv(path, on_bad_lines="skip") for path in self.data_path],
            ignore_index=True
        )
        transcript_df = transcript_df.dropna()
        transcript_df['number_of_words'] = transcript_df['line'].str.strip().str.split(" ")
        transcript_df['number_of_words'] = transcript_df['number_of_words'].apply(lambda x: len(x))
        transcript_df['eleven_res'] = 0
        transcript_df.loc[(transcript_df['name']=="Eleven")&(transcript_df['number_of_words']>=2),'eleven_res']=1
        
        indexes_to_take = list(transcript_df[(transcript_df['eleven_res'] == 1)&(transcript_df.index>0)].index)
        system_prompt = """ You are Eleven (or El for short), a character from the Netflix series Stranger Things. Your responses should reflect her personality and speech patterns \n"""
        prompts = []
        for ind in indexes_to_take:
            prompt = system_prompt
            prompt += transcript_df.iloc[ind -1]['line'] + '\n'
            prompt += transcript_df.iloc[ind]['line']
            prompts.append(prompt)
            
        df = pd.DataFrame({"prompt": prompts})
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def train(self,
          base_model_name_or_path,
          dataset,
          output_dir="./results",
          per_device_train_batch_size=1,
          gradient_accumulation_steps=1,
          optimizer="paged_adamw_32bit",
          save_steps=200,
          logging_steps=10,
          learning_rate=2e-4,
          max_grad_norm=0.3,
          max_steps=300,
          warmup_ratio=0.3,
          lr_scheduler_type="constant"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        toknizer = AutoTokenizer.from_pretrained(base_model_name_or_path)  # Kept typo
        toknizer.pad_token = toknizer.eos_token
        
        # Preprocess dataset
        def preprocess_function(examples):
            return toknizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)
        
        dataset = dataset.map(preprocess_function, batched=True)
        
        # LoRA config
        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optimizer,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            fp16=True,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
            report_to="none",
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            args=training_args,
        )
        trainer.train()
        
        # Save model
        trainer.model.save_pretrained("final_ckpt")
        toknizer.save_pretrained("final_ckpt")  # Kept typo
        
        # Flush memory
        del model, trainer
        gc.collect()
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            return_dict=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        
        toknizer = AutoTokenizer.from_pretrained(base_model_name_or_path)  # Kept typo
        model = PeftModel.from_pretrained(base_model, "final_ckpt")
        model.push_to_hub(self.model_path)
        toknizer.push_to_hub(self.model_path)  # Kept typo
        
        # Flush memory
        del model, base_model
        gc.collect()
        
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
            max_length=256,
            eos_token_id=terminator,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        output_message = output[0]['generated_text'][-1]
        return output_message