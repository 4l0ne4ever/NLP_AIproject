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
