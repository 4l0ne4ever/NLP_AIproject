# Stranger Things AI Analysis Suite - Project Guide

Comprehensive guide for understanding, developing, and deploying the Stranger Things AI Analysis Suite.

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Component Details](#component-details)
4. [Deployment Workflow](#deployment-workflow)
5. [Development Guide](#development-guide)
6. [Configuration Reference](#configuration-reference)
7. [Troubleshooting](#troubleshooting)

## Project Overview

### Purpose
This project creates an intelligent NLP application that analyzes Stranger Things content using custom-trained AI models with smart fallback capabilities to HuggingFace models.

### Key Innovations
- **Smart Model Management**: Automatically loads custom models from S3, falls back to HuggingFace when needed
- **Configurable Fallback Announcements**: Notifies users when fallback models are being used
- **Comprehensive Training Pipeline**: End-to-end training from transcript data to deployed models
- **AWS Integration**: Full cloud deployment with EC2 training and S3 model storage

## System Architecture

### High-Level Flow
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Data Processing │ ──► │   Model Training │ ──► │  Model Storage  │
│  (Transcripts)  │    │   (EC2 Instances) │    │      (S3)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ User Interface  │ ◄── │ Model Management │ ◄── │ Fallback Models │
│   (Gradio)      │    │    (S3 + HF)     │    │ (HuggingFace)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Component Interaction
- **Training Pipeline**: Processes transcript data → Trains models → Uploads to S3
- **Model Manager**: Checks S3 for trained models → Falls back to HuggingFace if needed
- **Gradio App**: Provides user interface → Uses model manager for inference
- **AWS Orchestrator**: Manages EC2 instances and S3 storage

## Component Details

### Core Application (`gradio_app_v2.py`)
- **Purpose**: Main user interface with fallback announcement system
- **Features**:
  - Character chatbots (LLaMA and Qwen)
  - Theme classification
  - Character network analysis
  - Location classification
  - Real-time model status display
  - Configurable fallback announcements

### Training Pipeline (`training_pipeline.py`)
- **Purpose**: End-to-end model training and S3 upload
- **Process**:
  1. Loads transcript CSV files from `data/transcripts/` (locally) or downloads them from S3 when invoked by the EC2 orchestrator using `--s3-transcripts`
  2. Preprocesses 10,924+ dialogue samples and uploads the processed dataset to S3 (`data/processed/{run_id}/training_data.json`)
  3. Fine-tunes models using LoRA (with logs to `output_dir/logs`)
  4. Uploads checkpoints (`checkpoints/{model_type}/{run_id}/`), logs (`logs/{model_type}/{run_id}/`), and the merged final model (`models/trained/{model_type}/{run_id}/`) to S3; updates `latest.json`
- **Supported Models**: LLaMA, Qwen

### AWS Deployment (`deploy_aws.py`)
- **Purpose**: Complete AWS deployment orchestration
- **Commands**:
  - `init`: Initialize AWS configuration
  - `upload-data`: Upload training data to S3
  - `train <model>`: Launch EC2 training instance
  - `deploy-gradio`: Deploy Gradio app to EC2
  - `status`: Show overall project status
  - `list-instances`: List running instances

### AWS Infrastructure (`aws/`)
- **`ec2_orchestrator.py`**: Manages EC2 instance lifecycle
- **`ec2_manager.py`**: Low-level EC2 operations
- **`storage.py`**: S3 storage management
- **`config.py`**: AWS configuration management

### Model Components
- **`character_chatbot/`**: LLaMA and Qwen chatbot implementations
- **`theme_classifier/`**: Zero-shot theme classification
- **`character_network/`**: Relationship network analysis
- **`text_classification/`**: Location classification

## Deployment Workflow

### Phase 1: Initial Setup
```bash
# 1. Environment setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. AWS configuration
aws configure
python deploy_aws.py init

# 3. Environment variables
cp .env.example .env
# Edit .env with HuggingFace token
```

### Phase 2: Data and Training
```bash
# 1. Upload training data (transcript files)
python deploy_aws.py upload-data

# 2. Train models (automatically saves to S3)
python deploy_aws.py train llama    # ~2-3 hours
python deploy_aws.py train qwen     # ~2-3 hours
```

### Phase 3: Application Deployment
```bash
# 1. Deploy Gradio app to EC2
python deploy_aws.py deploy-gradio

# 2. Monitor deployment
python deploy_aws.py status

# 3. Access application
# URL will be provided in deployment output
```

### Phase 4: Local Development (Optional)
```bash
# Run locally with fallback models during training
python gradio_app_v2.py
```

## Development Guide

### Adding New Models

1. **Create Model Class**:
```python
# In appropriate module directory
class NewModelChatbot:
    def __init__(self, model_path, huggingface_token):
        # Implementation
        pass
    
    def chat(self, message, history):
        # Implementation
        return response
```

2. **Update Configuration**:
```python
# In config.py
FALLBACK_MODELS["new_model"] = "huggingface/model-name"
S3_MODEL_PATHS["new_model"] = "models/trained/new_model/"
```

3. **Update Training Pipeline**:
```python
# In training_pipeline.py
elif self.model_type == "new_model":
    model = NewModelChatbot(
        self.base_model_name or "default/model",
        huggingface_token=os.getenv('huggingface_token')
    )
```

### Modifying Fallback Behavior

1. **Update Configuration**:
```python
# In config.py
FALLBACK_CONFIG = {
    "announce_fallback": True,
    "fallback_message": "Custom message for {model_type}",
    "log_fallbacks": True,  # New option
    "max_logged_messages": 5  # New option
}
```

2. **Update Announcement Function**:
```python
# In gradio_app_v2.py
def announce_fallback(model_type):
    # Custom logic here
    pass
```

### Adding New UI Features

1. **Create Processing Function**:
```python
def new_feature_function(input_data):
    # Process input
    return result
```

2. **Add to Gradio Interface**:
```python
# In gradio_app_v2.py main() function
with gr.TabItem("New Feature"):
    with gr.Row():
        input_component = gr.Textbox()
        output_component = gr.Textbox()
        process_btn = gr.Button("Process")
    
    process_btn.click(new_feature_function, inputs=input_component, outputs=output_component)
```

## Configuration Reference

### Main Configuration (`config.py`)

```python
# S3 Settings
S3_BUCKET_NAME = "your-bucket-name"
S3_REGION = "us-east-1"
LOCAL_MODEL_CACHE = "/tmp/stranger_things_models"

# Model Paths
S3_MODEL_PATHS = {
    "llama": "models/trained/llama/",
    "qwen": "models/trained/qwen/"
}

# Fallback Models
FALLBACK_MODELS = {
    "llama": "christopherxzyx/StrangerThings_Llama-3-8B_v3",
    "qwen": "christopherxzyx/StrangerThings_Qwen-3-4B"
}

# Fallback Configuration
FALLBACK_CONFIG = {
    "announce_fallback": True,
    "fallback_message": "Notice: Using HuggingFace fallback for {model_type}"
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 4,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "max_length": 512,
    "warmup_steps": 100
}

# Gradio Configuration
GRADIO_CONFIG = {
    "server_name": "0.0.0.0",
    "server_port": 7860,
    "share": True,
    "debug": True
}
```

### AWS Configuration (`aws_config.yaml`)

```yaml
ec2:
  ami_id: ami-0c02fb55956c7d316
  instance_type: g4dn.xlarge
  key_pair_name: your-key-pair
  region: us-east-1
  use_spot_instances: false
  max_spot_price: 0.5
  volume_size: 100

s3:
  bucket_name: your-bucket-name
  region: us-east-1
  model_prefix: models/
  training_data_prefix: data/training/

training:
  base_model_llama: meta-llama/Llama-3.2-3B-Instruct
  base_model_qwen: Qwen/Qwen2.5-3B-Instruct
  batch_size: 1
  learning_rate: 0.0002
  max_steps: 1000
```

### Environment Variables (`.env`)

```env
# Required
huggingface_token=your_hf_token_here

# Optional (if not using aws configure)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Optional overrides
S3_BUCKET_NAME=custom-bucket-name
```

## Monitoring and Debugging

### Training Monitoring

```bash
# Check training progress
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
tail -f /home/ubuntu/stranger-things-nlp/training.log

# Monitor system resources
htop
nvidia-smi
```

### Application Monitoring

```bash
# Check Gradio app logs
ssh -i ~/.ssh/your-key.pem ec2-user@<instance-ip>
tail -f /home/ec2-user/stranger-things-nlp/gradio.log

# Check model status in app
# Visit the "Model Status" section in Gradio interface
```

### S3 Model Verification

```bash
# List trained models
aws s3 ls s3://your-bucket/models/trained/ --recursive

# Check model info
aws s3 cp s3://your-bucket/models/trained/llama/latest.json .
cat latest.json
```

## Troubleshooting

### Common Issues

#### 1. Training Fails to Start
```bash
# Check EC2 instance status
python deploy_aws.py list-instances

# SSH into instance and check logs
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
ls -la /home/ubuntu/stranger-things-nlp/
cat /home/ubuntu/stranger-things-nlp/training.log
```

#### 2. Models Not Loading from S3
```bash
# Verify S3 permissions
aws s3 ls s3://your-bucket/models/trained/

# Check model info files
aws s3 cp s3://your-bucket/models/trained/llama/latest.json .
```

#### 3. Fallback Models Not Working
```bash
# Verify HuggingFace token
cat .env | grep huggingface_token

# Test token manually
python -c "from huggingface_hub import login; login('your_token_here')"
```

#### 4. Gradio App Not Accessible
```bash
# Check security group settings
# Ensure port 7860 is open in EC2 security group

# Check if app is running
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
ps aux | grep python
```

### Performance Optimization

#### Training Performance
- Use GPU instances (g4dn.xlarge recommended)
- Adjust batch size based on available memory
- Enable gradient checkpointing for larger models
- Use spot instances for cost savings

#### Inference Performance
- Cache models locally to reduce S3 download times
- Use smaller instances for hosting (t3.medium sufficient)
- Enable model quantization if supported
- Implement response caching for common queries

### Cost Optimization

#### Training Costs
- Use spot instances (up to 70% savings)
- Terminate instances when not in use
- Monitor training progress to avoid overtraining
- Use appropriate instance sizes

#### Storage Costs
- Use S3 Intelligent Tiering
- Clean up old model versions periodically
- Compress model files where possible
- Use lifecycle policies for automated cleanup

## Additional Resources

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Gradio Documentation](https://gradio.app/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Need Help?** Check the troubleshooting section or review the component-specific documentation in each module directory.

# Stranger Things NLP - Complete Project Guide

**Comprehensive Guide for Character Chatbot Development and AWS Deployment**

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Components](#architecture--components)
3. [Environment Setup](#environment-setup)
4. [Local Development](#local-development)
5. [AWS Deployment](#aws-deployment)
6. [Training Models](#training-models)
7. [Production Deployment](#production-deployment)
8. [Monitoring & Management](#monitoring--management)
9. [Troubleshooting](#troubleshooting)
10. [Cost Management](#cost-management)
11. [Model Evaluation Results](#model-evaluation-results)

---

## Project Overview

### About This Project
This is a comprehensive NLP project that creates character chatbots from the Netflix series Stranger Things. The system combines multiple machine learning approaches including named entity recognition, text classification, network analysis, and large language model fine-tuning to create an interactive character chatbot experience.

**Key Features:**
- Interactive character chatbots (Eleven, Mike, Dustin, etc.)
- Theme classification (Hope, Despair, Trust, Isolation, Rebellion, Danger, Love, Guilt)
- Character relationship network analysis
- Location classification system
- Production-ready AWS deployment
- Real-time Gradio web interface

**Technologies Used:**
- **Python**: Primary programming language
- **Scrapy**: Web data scraping
- **SpaCy**: Text processing and NER
- **Transformers**: LLM integration
- **Gradio**: User interface
- **AWS**: Cloud infrastructure (EC2, S3, IAM)

### Data Sources
- **Subtitles**: https://subtitlist.com/subs/stranger-things
- **Transcripts**: https://www.scribd.com/document/614545503/Stranger-Things-Transcript-101
- **Classification Dataset**: https://strangerthings.fandom.com/wiki/Stranger_Things_Wiki

---

## Architecture & Components

### Core Components Architecture

1. **Data Processing Pipeline**: 
   - `utils/data_loader.py` handles subtitle/script parsing from .srt files
   - Structured DataFrames with season/episode metadata

2. **NLP Analysis Layer**:
   - **Theme Classifier** (`theme_classifier/`): Zero-shot classification with DistilBART
   - **Character Network** (`character_network/`): NER with SpaCy + PyVis visualizations
   - **Text Classification** (`text_classification/`): DistilBERT fine-tuning for locations

3. **LLM Integration Layer** (`character_chatbot/`):
   - **Llama-based Chatbot**: Fine-tunes Llama 3.2-3B with LoRA adapters
   - **Qwen-based Chatbot**: Alternative using Qwen models
   - Conversational memory and character-specific system prompts

4. **Web Interface**: 
   - `gradio_app.py`: Unified interface orchestrating all components

5. **AWS Infrastructure**:
   - **S3 Storage**: Training data, models, results
   - **EC2 GPU Instances**: Model training
   - **EC2 Web Server**: Public Gradio hosting
   - **Automated Deployment**: CLI tools for management

### Data Flow Architecture
```
Raw Scripts (.srt) → Data Loader → Structured DataFrames
                                          ↓
Theme Classifier ← Processed Text → NER → Character Networks
                                          ↓
Location Classifier ← Text Features → Training Data → Fine-tuned Models
                                          ↓
Character Dialogue → LLM Fine-tuning → Character Chatbots → Gradio UI
```

### AWS Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   S3 Storage    │    │  EC2 Training   │    │  EC2 Gradio     │
│                 │    │                 │    │                 │
│ • Training Data │◄──►│ • GPU Instance  │    │ • Web Server    │
│ • Models        │    │ • LLaMA/Qwen    │    │ • Public Access │
│ • Results       │    │ • Auto-scaling  │    │ • Load Models   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Model Endpoints
- **Theme Classification**: `valhalla/distilbart-mnli-12-3`
- **Text Classification**: `distilbert/distilbert-base-uncased`
- **Character Chatbots**:
  - `christopherxzyx/StrangerThings_Llama-3-8B_v3`
  - `christopherxzyx/StrangerThings_Qwen-3-4B`
- **Base Models**: `meta-llama/Llama-3.2-3B-Instruct`

---

## Environment Setup

### Prerequisites
- **macOS** (this guide is for Mac users)
- **Python 3.8+**
- **Git**
- **AWS Account** with billing configured
- **HuggingFace Token** for model access

### Local Setup
```bash
# Navigate to project directory
cd "/Users/duongcongthuyet/Downloads/workspace/AI /project"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export huggingface_token="your_token_here"
# or create .env file: huggingface_token=your_token_here
```

### AWS Setup
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
rm AWSCLIV2.pkg

# Configure AWS credentials
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1), Output format (json)

# Verify setup
aws sts get-caller-identity
```

---

## Local Development

### Running the Application
```bash
# Launch main Gradio application
python gradio_app.py
```

### Running Individual Components
```bash
# Theme classification
python -c "from theme_classifier import ThemeClassifier; tc = ThemeClassifier(['Hope', 'Despair', 'Trust'])"

# Character network analysis
python -c "from character_network import NamedEntityRecognizer; ner = NamedEntityRecognizer()"

# Text classification
python -c "from text_classification import LocationClassifier; lc = LocationClassifier('model_path', 'data_path')"
```

### Testing and Evaluation
```bash
# Evaluate chatbot models
python character_chatbot/evaluate.py

# Calculate BLEU scores
python character_chatbot/bleuScore.py
```

---

## AWS Deployment

### Initial Setup (One-time)

#### Step 1: Initialize AWS Configuration
```bash
source venv/bin/activate
python3 deploy_aws.py init
```

This creates `aws_config.yaml` with your customized settings.

#### Step 2: Create IAM Role and Instance Profile
```bash
# Create IAM role for EC2-S3 access
aws iam create-role --role-name EC2-S3-Access --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

# Attach policies
aws iam attach-role-policy --role-name EC2-S3-Access --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-role-policy --role-name EC2-S3-Access --policy-arn arn:aws:iam::aws:policy/AmazonEC2ReadOnlyAccess

# Create instance profile
aws iam create-instance-profile --instance-profile-name EC2-S3-Access
aws iam add-role-to-instance-profile --instance-profile-name EC2-S3-Access --role-name EC2-S3-Access
```

#### Step 3: Create EC2 Key Pair
```bash
aws ec2 create-key-pair --key-name stranger-things-key-dct --query 'KeyMaterial' --output text > ~/.ssh/stranger-things-key-dct.pem
chmod 400 ~/.ssh/stranger-things-key-dct.pem
```

#### Step 4: Upload Training Data
```bash
python3 deploy_aws.py upload-data
```

### Configuration Options

Key settings in `aws_config.yaml`:
```yaml
ec2:
  instance_type: g4dn.xlarge      # GPU for training
  key_pair_name: stranger-things-key-dct
  region: us-east-1
  use_spot_instances: false       # Use on-demand for reliability
  max_spot_price: 0.5

s3:
  bucket_name: stranger-things-nlp-duongcongthuyet
  region: us-east-1

training:
  base_model_llama: meta-llama/Llama-3.2-3B-Instruct
  base_model_qwen: Qwen/Qwen2.5-3B-Instruct
  batch_size: 1
  learning_rate: 2e-4
  max_steps: 1000
  fp16: true
  gradient_checkpointing: true
  lora_r: 64
  lora_alpha: 16
  lora_dropout: 0.1

gradio:
  port: 7860
  host: 0.0.0.0
  enable_auth: true
```

---

## Training Models

### Start Training
```bash
source venv/bin/activate

# Train LLaMA model (recommended first)
python3 deploy_aws.py train llama

# Train Qwen model
python3 deploy_aws.py train qwen
```

### Training Process
1. **Instance Launch** (2-3 minutes)
   - Launches g4dn.xlarge GPU instance
   - Installs CUDA and Python dependencies
   - Sets up training environment

2. **Code Deployment** (1-2 minutes)
   - Uploads project code to EC2
   - Installs Python requirements
   - Downloads training data from S3

3. **Training Process** (1-3 hours)
   - Fine-tunes model with your dialogue data
   - Monitors training progress
   - Uses LoRA for efficient parameter updates

4. **Model Upload** (5-10 minutes)
   - Pushes trained model to HuggingFace Hub
   - Backs up model to S3
   - Saves training results and metrics

### Monitor Training
```bash
# Check training status
python3 deploy_aws.py list-instances

# SSH into training instance
ssh -i ~/.ssh/stranger-things-key-dct.pem ubuntu@<INSTANCE_IP>

# On training instance:
tail -f training.log    # Watch progress
nvidia-smi             # Check GPU usage
htop                   # System resources
```

---

## Production Deployment

### Deploy Gradio App
```bash
# Deploy app publicly
python3 deploy_aws.py deploy-gradio

# Get app URL
python3 deploy_aws.py status
```

**What happens:**
1. Launches t3.medium instance (~$0.02/hour)
2. Uploads Gradio app code
3. Starts app on port 7860
4. App accessible at `http://<PUBLIC_IP>:7860`

### Security Features
- **IAM Roles**: Proper EC2-S3 permissions
- **Security Groups**: Controlled network access
- **Key Management**: Secure SSH key handling
- **Authentication**: Optional Gradio authentication

---

## Monitoring & Management

### Daily Management
```bash
source venv/bin/activate

# Check everything
python3 deploy_aws.py status

# List running instances
python3 deploy_aws.py list-instances

# Terminate instance (save money!)
python3 deploy_aws.py terminate i-1234567890abcdef0
```

### S3 Data Structure
```
stranger-things-nlp/
├── data/
│   ├── training/           # Training datasets
│   ├── transcripts/        # Character dialogue data
│   └── subtitles/          # Raw subtitle files
├── models/
│   ├── llama-chatbot/      # Trained LLaMA model
│   ├── qwen-chatbot/       # Trained Qwen model
│   └── location-classifier/ # Location classification model
├── results/
│   ├── training_results.json
│   ├── evaluation_metrics.json
│   └── bleu_scores.json
├── checkpoints/            # Training checkpoints
└── logs/                   # Training logs
```

### Log Locations on EC2
```
/home/ubuntu/stranger-things-nlp/
├── training.log          # Training output
├── gradio.log           # Gradio app logs
├── setup.log            # Initial setup logs
└── gradio_setup.log     # Gradio setup logs
```

---

## Troubleshooting

### Common Issues

#### AWS Permission Denied
**Error:** `AccessDenied` when running commands

**Solution:**
1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam/)
2. Add these policies to your user:
   - `AmazonEC2FullAccess`
   - `AmazonS3FullAccess`
   - `IAMFullAccess`

#### EC2 Instance Launch Failed
**Error:** `Invalid key pair` or `Security group` errors

**Solution:**
```bash
# Recreate key pair
aws ec2 delete-key-pair --key-name stranger-things-key-dct
aws ec2 create-key-pair --key-name stranger-things-key-dct --query 'KeyMaterial' --output text > ~/.ssh/stranger-things-key-dct.pem
chmod 400 ~/.ssh/stranger-things-key-dct.pem
```

#### Training Out of Memory
**Error:** `CUDA out of memory`

**Solution:**
- Reduce `batch_size` in `aws_config.yaml`
- Enable `gradient_checkpointing: true`
- Use `fp16: true` for memory optimization

#### Gradio App Not Accessible
**Check:**
1. Instance running: `python3 deploy_aws.py list-instances`
2. Security group allows port 7860
3. App started: SSH and check `tail -f gradio.log`

### Debug Commands
```bash
# SSH into training instance
ssh -i ~/.ssh/stranger-things-key-dct.pem ubuntu@<instance-ip>

# Check training logs
tail -f training.log

# Check GPU usage
nvidia-smi

# Check system resources
htop
```

---

## Cost Management

### Cost Estimates

| Resource | Instance/Type | Cost/Hour | Monthly* |
|----------|---------------|-----------|----------|
| Training | g4dn.xlarge (on-demand) | ~$1.00 | ~$20** |
| Gradio App | t3.medium | ~$0.02 | ~$15 |
| S3 Storage | Standard | - | ~$2/10GB |

*24/7 operation **Training typically runs 1-3 hours per model

### Cost Optimization

#### Instance Management
```bash
# Daily routine - check what's running
python3 deploy_aws.py list-instances

# Terminate training instances when done
python3 deploy_aws.py terminate i-1234567890abcdef0

# Emergency - stop all instances
aws ec2 describe-instances --query 'Reservations[*].Instances[*].InstanceId' --output text | xargs aws ec2 terminate-instances --instance-ids
```

#### S3 Storage Classes
- Use **Standard** for frequently accessed data
- Move old training data to **IA** (Infrequent Access)
- Archive old models to **Glacier**

#### Monitoring Costs
- Set up AWS billing alerts
- Use AWS Cost Explorer
- Tag resources for cost tracking

### Auto-Shutdown
Add to training scripts:
```bash
# Auto-shutdown after training
shutdown -h +30  # Shutdown in 30 minutes
```

---

## Model Evaluation Results

### Performance Comparison

#### Accuracy Results
- **Llama-3.2-3B-Instruct**: 60% accuracy
- **Qwen3-4B**: 76% accuracy

#### Response Quality Analysis
**Sample Question**: "Do you like waffles more than anything?"

**Llama Response** (Score: 1.411):
```
"excitedly Waffles! nods enthusiastically Yes, I like waffles more than anything. 
Especially with butter and syrup. smiles slightly The Wheeler kids make them for me sometimes."
```

**Qwen Response** (Score: 0.387):
```
"Waffles? Oh, waffles… I don't really like things like that. I don't like food. 
I don't like anything. I don't like being normal…"
```

**Standard Response** (Score: 0.875):
```
"Waffles with butter and syrup are my favorite."
```

### Model Analysis

#### Meta-Llama/Llama-3.2-3B-Instruct
**Strengths:**
- Optimized for conversation with human-annotated training
- Reproduces Eleven's concise tone and emotions effectively
- Large context window (128,000 tokens) for conversation consistency
- Less restrictive content policy for creative scenarios
- Accurate character context (Wheeler house reference)

**Weaknesses:**
- Lower overall accuracy (60%) on general knowledge
- May lack complex film details or pop culture knowledge

#### Qwen/Qwen3-4B
**Strengths:**
- Higher accuracy (76%) for detail reproduction
- 23% faster processing for real-time chatbot use
- Better multilingual support (29+ languages including Vietnamese)
- Strong keyword generation for semantic matching

**Weaknesses:**
- Weaker pop culture knowledge (50/100 vs Llama's 77.9)
- Less accurate character tone and personality capture
- Stricter content filtering may limit creativity
- May generate keyword-rich but contextually incorrect responses

### Evaluation with gte-multilingual-reranker-base
- Prioritizes responses with relevant keywords ("waffles," "butter," "syrup")
- May not evaluate character tone or personality nuances well
- Can favor semantically rich but character-inaccurate responses

### Recommendation
**For Stranger Things Character Chatbots:**
- **Llama-3.2-3B-Instruct** is better suited for character representation
- More natural dialogue flow and emotional expression
- Better context awareness and character consistency
- Despite lower general accuracy, performs excellently for character-specific interactions

---

## Quick Reference

### Setup (One-time)
```bash
cd "/Users/duongcongthuyet/Downloads/workspace/AI /project"
source venv/bin/activate
python3 deploy_aws.py init
python3 deploy_aws.py upload-data
```

### Training Models
```bash
source venv/bin/activate
python3 deploy_aws.py train llama    # Train LLaMA
python3 deploy_aws.py train qwen     # Train Qwen
python3 deploy_aws.py list-instances # Check status
```

### Deploy Gradio App
```bash
source venv/bin/activate
python3 deploy_aws.py deploy-gradio
python3 deploy_aws.py status         # Get app URL
```

### Daily Management
```bash
source venv/bin/activate
python3 deploy_aws.py status         # Check everything
python3 deploy_aws.py list-instances # List running instances
python3 deploy_aws.py terminate <id> # Terminate instance
```

### Emergency Commands
```bash
# Stop all instances
aws ec2 describe-instances --query 'Reservations[*].Instances[*].InstanceId' --output text | xargs aws ec2 terminate-instances --instance-ids

# Check AWS costs
open https://console.aws.amazon.com/billing/
```

**Always terminate training instances after completion to save money!**

---

## Additional Resources

- [HuggingFace Model Hub](https://huggingface.co/christopherxzyx)
- [AWS EC2 Pricing](https://aws.amazon.com/ec2/pricing/)
- [AWS S3 Pricing](https://aws.amazon.com/s3/pricing/)
- [Gradio Documentation](https://gradio.app/docs/)

---

**Ready to create your own Stranger Things character chatbot? Start with `python3 deploy_aws.py init`!**
