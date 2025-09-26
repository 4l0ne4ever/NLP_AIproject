# Configuration settings for the Gradio app

# S3 Settings
S3_BUCKET_NAME = "stranger-things-nlp-duongcongthuyet"  
S3_REGION = "us-east-1"  

# Model paths in S3
S3_MODEL_PATHS = {
    "llama": "models/trained/llama/",
    "qwen": "models/trained/qwen/",
    "location_classifier": "models/trained/location_classifier/"
}

# Local cache directory for downloaded models
LOCAL_MODEL_CACHE = "/tmp/stranger_things_models"

# HuggingFace fallback models
FALLBACK_MODELS = {
    "llama": "christopherxzyx/StrangerThings_Llama-3-8B_v3",
    "qwen": "christopherxzyx/StrangerThings_Qwen-3-4B"
}

# Training data paths in S3
S3_DATA_PATHS = {
    "subtitles": "data/training/Subtitles/",
    "processed_data": "data/processed/",
    "training_outputs": "data/training_outputs/"
}

# Default theme categories for classification
DEFAULT_THEMES = [
    "friendship",
    "mystery", 
    "supernatural",
    "coming of age",
    "family",
    "adventure",
    "horror",
    "romance",
    "betrayal",
    "sacrifice"
]

# Fallback model settings
FALLBACK_CONFIG = {
    "announce_fallback": True,  # Set to False to disable fallback announcements
    "fallback_message": "Notice: Using HuggingFace fallback model for {model_type}. Custom trained model not available."
}

# Gradio settings
GRADIO_CONFIG = {
    "server_name": "0.0.0.0",
    "server_port": 7860,
    "share": True,
    "debug": True,
    "show_error": True,
    "max_threads": 4
}

# Model training parameters
TRAINING_CONFIG = {
    "batch_size": 8,  # Increased for T4 GPU (16GB VRAM)
    "gradient_accumulation_steps": 4,  # Effective batch size = 8 * 4 = 32
    "learning_rate": 2e-4,  # Slightly higher for faster convergence
    "num_epochs": 2,  # Reduced for faster training
    "max_length": 256,  # Reduced for dialogue training and speed
    "warmup_steps": 50,  # Reduced warmup
    "max_steps": 800,  # ~2.5 epochs for good training (45-60 min)
    "logging_steps": 10,  # More frequent logging
    "save_steps": 50,  # More frequent checkpoints
    "eval_steps": 25,  # More frequent evaluation
    "fp16": True,  # Enable mixed precision for speed
    "dataloader_num_workers": 2,  # Parallel data loading
    "remove_unused_columns": False,
    "gradient_checkpointing": True,  # Memory optimization
    "optim": "adamw_torch_fused"  # Faster optimizer for GPU
}
