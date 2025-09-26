# Stranger Things AI Analysis Suite

A comprehensive NLP application that analyzes Stranger Things content using custom-trained AI models with intelligent fallback to HuggingFace models.

## Features

- **Character Chatbots**: Chat with Stranger Things characters using fine-tuned LLaMA and Qwen models
- **Theme Classification**: Zero-shot theme analysis of text content
- **Character Network Analysis**: Interactive relationship network visualization
- **Location Classification**: Classify text locations using fine-tuned models
- **Smart Model Management**: Automatic S3 model loading with HuggingFace fallbacks
- **Fallback Announcements**: Configurable notifications when using fallback models
- **AWS Deployment**: Complete EC2/S3 training and deployment pipeline

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio App    │────│   S3 Storage     │────│  EC2 Training   │
│  (User Interface│    │ (Trained Models) │    │   Instances     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                        │
         │              ┌────────┴────────┐               │
         └──────────────│ HuggingFace Hub │───────────────┘
                        │ (Fallback Models)│
                        └─────────────────┘
```

## Project Structure

```
├── gradio_app_v2.py          # Main Gradio application with fallback announcements
├── training_pipeline.py      # Model training and S3 upload pipeline
├── deploy_aws.py            # AWS deployment orchestrator
├── config.py                # Configuration settings
├── aws/                     # AWS infrastructure management
│   ├── config.py           # AWS configuration
│   ├── ec2_orchestrator.py  # EC2 instance management
│   ├── ec2_manager.py       # EC2 operations
│   └── storage.py          # S3 storage management
├── data/
│   └── transcripts/        # Training data (10,924+ dialogue samples)
├── character_chatbot/      # Character chatbot implementations
├── theme_classifier/       # Theme classification module
├── character_network/      # Character relationship analysis
└── text_classification/    # Location classification module
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install gradio boto3 transformers torch pandas numpy python-dotenv \
            nltk spacy scikit-learn matplotlib seaborn pyvis datasets \
            evaluate peft accelerate trl

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### 2. Configuration

```bash
# Configure AWS credentials
aws configure

# Set up environment variables
cp .env.example .env
# Edit .env with your HuggingFace token
```

### 3. Local Development

```bash
# Run Gradio app locally (uses HuggingFace fallbacks initially)
source venv/bin/activate
python gradio_app_v2.py
```

### 4. AWS Training & Deployment

```bash
# Initialize AWS infrastructure
python deploy_aws.py init

# Upload training data
python deploy_aws.py upload-data

# Train models (saves to S3 automatically)
python deploy_aws.py train llama
python deploy_aws.py train qwen

# Deploy production Gradio app
python deploy_aws.py deploy-gradio

# Monitor deployment
python deploy_aws.py status
```

## Configuration

### Fallback Announcements

Customize fallback behavior in `config.py`:

```python
FALLBACK_CONFIG = {
    "announce_fallback": True,  # Enable/disable announcements
    "fallback_message": "Notice: Using HuggingFace fallback for {model_type}"
}
```

### Model Paths

```python
# S3 model locations
S3_MODEL_PATHS = {
    "llama": "models/trained/llama/",
    "qwen": "models/trained/qwen/",
    "location_classifier": "models/trained/location_classifier/"
}

# HuggingFace fallback models
FALLBACK_MODELS = {
    "llama": "christopherxzyx/StrangerThings_Llama-3-8B_v3",
    "qwen": "christopherxzyx/StrangerThings_Qwen-3-4B"
}
```

## Training Data

The system uses **10,924+ dialogue samples** from Stranger Things transcripts:

- **Format**: CSV files with `name,line` columns
- **Source**: `data/transcripts/*.csv`
- **Coverage**: Seasons 1-4 dialogue
- **Characters**: Mike, Eleven, Dustin, Lucas, Hopper, Joyce, and more

## Model Information

### Character Chatbots
- **LLaMA Model**: Fine-tuned for character dialogue generation
- **Qwen Model**: Alternative architecture for character interactions
- **Training**: LoRA fine-tuning on character-specific dialogue

### Classification Models
- **Theme Classifier**: Zero-shot classification for content themes
- **Location Classifier**: Fine-tuned model for location prediction
- **Network Analysis**: Character relationship extraction and visualization

## Development

### Local Training

```bash
# Train specific model locally using local transcripts
python training_pipeline.py --model-type llama --transcripts-dir data/transcripts
python training_pipeline.py --model-type qwen --transcripts-dir data/transcripts

# Or, train by downloading transcripts from S3 (useful for parity with EC2)
python training_pipeline.py --model-type llama --s3-transcripts data/transcripts/
python training_pipeline.py --model-type qwen --s3-transcripts data/transcripts/
```

### AWS Management

```bash
# List running instances
python deploy_aws.py list-instances

# Check overall status
python deploy_aws.py status

# Terminate instance
python deploy_aws.py terminate <instance-id>
```

### SSH Access

```bash
# Access training instance
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
tail -f /home/ubuntu/stranger-things-nlp/training.log

# Access Gradio instance
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
tail -f /home/ubuntu/stranger-things-nlp/gradio.log
```

## Monitoring

### Model Status
- The Gradio interface shows current model status
- Fallback announcements indicate when using HuggingFace models
- S3 model info includes training metrics and timestamps

### Training Progress
- Monitor via SSH: `tail -f training.log`
- Check S3 for completed models
- View training metrics in model info JSON files

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Run the full pip install command
2. **AWS Permissions**: Ensure EC2, S3, and IAM permissions are configured
3. **Key Pair**: Create and configure EC2 key pair in AWS console
4. **Model Loading**: Check S3 bucket permissions and model file integrity
5. **Fallback Issues**: Verify HuggingFace token in `.env` file

### Performance Optimization

- **Instance Types**: Use g4dn.xlarge for training, t3.medium for hosting
- **Spot Instances**: Enable for cost savings during training
- **Model Caching**: Local model cache reduces S3 download times
- **Batch Size**: Adjust training batch size based on instance memory

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Stranger Things content © Netflix
- HuggingFace Transformers library
- Gradio framework for web interfaces
- AWS infrastructure services

# Stranger Things NLP - Character Chatbot Project

**Interactive Character Chatbots from Netflix's Stranger Things**

**Link to trained models:** https://huggingface.co/christopherxzyx

**[Complete Project Guide](PROJECT_GUIDE.md)** - Comprehensive documentation for setup, development, and deployment

**[AWS SageMaker Integration](SAGEMAKER_INTEGRATION.md)** - Production-ready ML platform with scalable training and deployment

---

## About this project

Problem:
In the entertainment industry, especially with series like Stranger Things, fans often want to engage more deeply with their favorite characters like Eleven, Mike, or Dustin.
However, merely watching the series or reading about it cannot provide the experience of direct interaction. The problem posed is:

- How to analyze text content from Stranger Things (subtitles, scripts, characters) to clearly understand the linguistic style and personality traits of a specific character.
- How to build a chatbot system that allows users to converse naturally with that character, accurately reflecting their language style and behavior.

Goals:

- Create a chatbot based on characters from Stranger Things, using Natural Language Processing (NLP) and Large Language Models (LLM), to offer a unique interactive experience.
- Applications: Satisfy fan interest in exploring and interacting, support pop culture research, or serve as an educational tool for studying NLP.

Main Idea:
The solution leverages the power of NLP and LLM to analyze text data from Stranger Things, extract linguistic/personality traits of characters, and integrate them into a chatbot
system capable of generating natural, in-character responses. The project combines modern tools such as Scrapy, SpaCy, Transformers, and Gradio to create a complete workflow from
data collection to user interface deployment.

Proposed Method:
The method is divided into key steps:

1. Data Collection:

- Use Scrapy to scrape data from the web (subtitles, transcripts, characters, locations) related to Stranger Things.
- Filter data relevant to the target character (e.g., Eleven, Dustin).

2. Character Trait Analysis:

- Use SpaCy for Named Entity Recognition (NER) and syntactic parsing to extract distinctive vocabulary, phrases, and speaking style.
- Apply a Text Classifier to categorize emotions or topics in the character’s dialogues.

3. Integration of Large Language Model (LLM):

- Use an LLM as the base for text generation.
- Provide contextual prompts or fine-tune the model using character-specific data to ensure appropriate responses.
- Evaluate using Alibaba-NLP/gte-multilingual-reranker-base

4. User Interaction Handling:

- Analyze user input via NLP to understand intent.
- Generate responses using the LLM and character profile.

5. Interface Deployment:

- Use Gradio to build a web interface, display the chatbot, and allow users to interact directly.

Technologies Used:
Python: Primary programming language.
Scrapy: For web data scraping.
SpaCy: For text processing and linguistic analysis.
Hugging Face Transformers: For LLM integration.
Gradio: For the user interface.

Advantages of the Solution:
High personalization: Chatbot accurately reflects the style of Stranger Things characters.
Flexibility: Can be applied to multiple characters in the series.
User-friendly: Gradio interface is intuitive and requires no technical knowledge from users.

---

## Quick Start

### Local Development
```bash
# Setup environment
source venv/bin/activate
export huggingface_token="your_token_here"

# Run the application
python gradio_app.py
```

### AWS Production Deployment (EC2)
```bash
# Initialize AWS setup
source venv/bin/activate
python3 deploy_aws.py init

# Upload data and start training
python3 deploy_aws.py upload-data
python3 deploy_aws.py train llama

# Deploy publicly
python3 deploy_aws.py deploy-gradio
```

### AWS SageMaker Deployment (Recommended for Production)
```bash
# Setup and deploy with SageMaker
cd sagemaker

# Build custom training container
cd docker && ./build_and_push.sh && cd ..

# Deploy infrastructure, train models, and create endpoints
python deploy.py --setup-infrastructure --train-all --deploy-all

# Launch SageMaker-enabled interface
python gradio_app.py
```

**For detailed instructions, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md) and [SAGEMAKER_INTEGRATION.md](SAGEMAKER_INTEGRATION.md)**

---

## Project Structure

```
project/
├── PROJECT_GUIDE.md        # Complete documentation
├── README.md              # This file
├── SAGEMAKER_INTEGRATION.md # AWS SageMaker integration guide
├── WARP.md               # Development guidelines
├── gradio_app.py         # Main web interface
├── deploy_aws.py         # AWS deployment CLI
├── aws_config.yaml       # AWS configuration
├── requirements.txt      # Dependencies
├── venv/                # Python environment
├── sagemaker/           # AWS SageMaker integration
├── aws/                 # AWS infrastructure code
├── character_chatbot/   # Chatbot models
├── theme_classifier/    # Theme analysis
├── character_network/   # Network analysis
├── text_classification/ # Location classifier
├── utils/              # Utilities
├── data/               # Training data
└── stubs/              # Output results
```
