# Stranger Things AI Analysis Suite v2 - Gradio Application

This document describes the enhanced Gradio application with intelligent fallback model management and configurable announcements.

## Overview

The Gradio application (`gradio_app_v2.py`) serves as the main user interface for the Stranger Things AI Analysis Suite. It provides a comprehensive web interface for interacting with trained AI models while gracefully handling fallback scenarios.

## Key Features

### Smart Model Management
- **Automatic S3 Detection**: Checks for trained models in S3 bucket
- **Intelligent Fallback**: Seamlessly falls back to HuggingFace models when S3 models unavailable
- **Configurable Announcements**: Notifies users when fallback models are being used
- **Real-time Status**: Displays current model status and training information

### Core Functionality

1. **Character Chatbots**
   - LLaMA-based character interactions
   - Qwen-based character conversations
   - Trained on 10,924+ dialogue samples

2. **Theme Classification**
   - Zero-shot theme analysis
   - Customizable theme categories
   - Visual results with bar charts

3. **Character Network Analysis**
   - Interactive relationship visualization
   - Named entity recognition
   - Network graph generation

4. **Location Classification**
   - Fine-tuned location prediction
   - Custom model support
   - Text-based classification

## Configuration

### Fallback Settings

Configure fallback behavior in `config.py`:

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

# Fallback models
FALLBACK_MODELS = {
    "llama": "christopherxzyx/StrangerThings_Llama-3-8B_v3",
    "qwen": "christopherxzyx/StrangerThings_Qwen-3-4B"
}
```

### Gradio Settings

```python
GRADIO_CONFIG = {
    "server_name": "0.0.0.0",
    "server_port": 7860,
    "share": True,
    "debug": True,
    "show_error": True,
    "max_threads": 4
}
```

## Model Loading Process

### Initialization Sequence

1. **S3 Model Check**: Application checks S3 for trained models
2. **Model Info Retrieval**: Downloads model metadata from `latest.json`
3. **Model Download**: Downloads model files to local cache if available
4. **Fallback Decision**: Falls back to HuggingFace if S3 models unavailable
5. **Announcement**: Logs fallback usage if configured

### Fallback Announcement System

```python
def announce_fallback(model_type):
    """Announce when falling back to HuggingFace model"""
    if FALLBACK_CONFIG.get("announce_fallback", True):
        message = FALLBACK_CONFIG.get("fallback_message").format(model_type=model_type)
        print(message)
        fallback_messages.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": model_type,
            "message": message
        })
```

## User Interface Components

### Model Status Section
- Current model status (S3 or Fallback)
- Training timestamp and accuracy metrics
- Fallback announcement status
- Recent fallback messages log

### Theme Classification Interface
- Text input for content analysis
- Customizable theme categories
- Visual results with bar charts
- Save functionality for results

### Character Network Interface
- Subtitle file upload
- Interactive network visualization
- Character relationship mapping
- Export capabilities

### Text Classification Interface
- Location prediction functionality
- Model path configuration
- Training data integration
- Classification results display

### Character Chatbot Interfaces
- **LLaMA Chatbot**: Character-based conversations
- **Qwen Chatbot**: Alternative model interactions
- Chat history management
- Response generation controls

## Running the Application

### Local Development

```bash
# Activate virtual environment
source venv/bin/activate

# Run application
python gradio_app_v2.py

# Access at http://localhost:7860
```

### Production Deployment

```bash
# Deploy to AWS EC2
python deploy_aws.py deploy-gradio

# Monitor deployment
python deploy_aws.py status
```

## Monitoring and Debugging

### Status Information

The application provides comprehensive status information:
- Model loading status
- Fallback announcements log
- Training metrics and timestamps
- S3 connectivity status

### Debug Mode

Enable debug mode for detailed logging:

```python
GRADIO_CONFIG = {
    "debug": True,
    "show_error": True
}
```

### Log Files

- **Console Output**: Real-time model loading and fallback announcements
- **Gradio Logs**: Application-specific logging
- **Model Status**: Displayed in web interface

## Customization

### Adding New Models

1. **Update Configuration**:
```python
FALLBACK_MODELS["new_model"] = "huggingface/model-name"
S3_MODEL_PATHS["new_model"] = "models/trained/new_model/"
```

2. **Create Model Interface**:
```python
def new_model_function(message, history):
    # Implementation
    return response
```

3. **Add to UI**:
```python
with gr.TabItem("New Model"):
    new_interface = gr.ChatInterface(new_model_function)
```

### Modifying Themes

Update default themes in `config.py`:

```python
DEFAULT_THEMES = [
    "friendship",
    "mystery",
    "supernatural",
    "coming of age",
    # Add custom themes
]
```

## Performance Considerations

### Model Caching
- Models cached locally to reduce S3 download times
- Cache location: `/tmp/stranger_things_models`
- Automatic cache management

### Resource Usage
- Memory requirements depend on loaded models
- CPU usage optimized for inference
- GPU acceleration when available

### Scaling
- Supports multiple concurrent users
- Thread management via `max_threads` setting
- Model sharing across requests

## Security

### Environment Variables
- HuggingFace token stored in `.env` file
- AWS credentials via standard AWS configuration
- Sensitive data not logged or displayed

### Access Control
- Optional authentication can be enabled
- IP-based access restrictions possible
- HTTPS support available

## Troubleshooting

### Common Issues

1. **Models Not Loading**
   - Check S3 bucket permissions
   - Verify model file integrity
   - Check internet connectivity for HuggingFace fallback

2. **Fallback Not Working**
   - Verify HuggingFace token in `.env`
   - Check model names in configuration
   - Monitor console for error messages

3. **UI Not Responding**
   - Check Gradio configuration
   - Monitor system resources
   - Review application logs

### Performance Issues

1. **Slow Model Loading**
   - Check S3 region and network speed
   - Monitor local cache usage
   - Consider model compression

2. **High Memory Usage**
   - Adjust model parameters
   - Monitor concurrent users
   - Consider model quantization

## Integration

### AWS Integration
- Seamless S3 model loading
- EC2 deployment support
- CloudWatch logging integration

### External APIs
- HuggingFace Hub integration
- Custom model endpoint support
- Third-party service integration

### Data Pipeline
- Training data integration
- Model versioning support
- Automated model updates

This enhanced Gradio application provides a robust, user-friendly interface for the Stranger Things AI Analysis Suite with intelligent model management and comprehensive fallback capabilities.

# Stranger Things AI Analysis Suite v2

## Overview

This is an enhanced version of your Stranger Things AI analysis application that integrates with AWS S3 for dynamic model loading and provides a clean, icon-free interface. The app automatically loads trained models from S3 when available and falls back to HuggingFace models.

## New Features

- **Dynamic Model Loading**: Automatically loads your custom trained models from S3
- **Clean Interface**: Removed all icons for a cleaner, text-based UI
- **Model Status Dashboard**: Real-time display of model versions and training info
- **Fallback System**: Uses HuggingFace models when S3 models aren't available
- **Improved Configuration**: Centralized configuration system
- **Training Pipeline**: Complete pipeline for training and uploading models

## Files Structure

```
project/
├── gradio_app_v2.py          # Enhanced Gradio app with S3 integration
├── config.py                 # Configuration settings
├── deploy_gradio_v2.py       # Deployment and setup script
├── training_pipeline.py      # Training pipeline for model management
├── gradio_app.py            # Original Gradio app (kept for reference)
├── .env                     # Environment variables (created by deploy script)
└── README_gradio_v2.md      # This documentation
```

## Setup Instructions

### 1. Prerequisites

Make sure you have the following installed:
- Python 3.8 or higher
- AWS CLI configured with appropriate permissions
- Required Python packages (will be checked by deployment script)

### 2. Quick Setup

Run the deployment script to set up everything automatically:

```bash
python deploy_gradio_v2.py
```

This script will:
- Check all dependencies
- Verify AWS configuration
- Create `.env` file template
- Set up S3 bucket structure
- Create model info templates
- Test the Gradio app

### 3. Manual Setup (Alternative)

If you prefer manual setup:

1. **Install dependencies**:
   ```bash
   pip install gradio boto3 transformers torch pandas numpy python-dotenv
   ```

2. **Configure AWS credentials**:
   ```bash
   aws configure
   ```

3. **Create `.env` file**:
   ```
   huggingface_token=your_huggingface_token_here
   ```

4. **Update configuration**:
   Edit `config.py` to set your S3 bucket name and other settings.

### 4. Configuration

Edit `config.py` to customize:
- S3 bucket name and region
- Model paths in S3
- Default themes for analysis
- Training parameters
- Gradio server settings

## Usage

### Running the Application

```bash
python gradio_app_v2.py
```

The app will be available at `http://0.0.0.0:7860`

### Training New Models

Use the training pipeline to train and upload models:

```bash
# Train LLama model
python training_pipeline.py --model-type llama --subtitles-dir /path/to/subtitles

# Train Qwen model with custom base model
python training_pipeline.py --model-type qwen --base-model Qwen/Qwen-1_8B-Chat --subtitles-dir /path/to/subtitles

# Train using S3 data
python training_pipeline.py --model-type llama --s3-subtitles data/training/Subtitles/
```

## Features

### 1. Model Status Dashboard
- Shows which models are loaded from S3 vs HuggingFace
- Displays training dates and accuracy metrics
- Refresh button to reload models from S3

### 2. Theme Classification
- Analyze themes in subtitle files
- Pre-populated with common Stranger Things themes
- Visual bar chart output

### 3. Character Network Analysis
- Generate interactive character relationship networks
- Named entity recognition on dialogue
- HTML-based network visualization

### 4. Location Classification
- Classify text locations using fine-tuned models
- Supports custom model paths
- Integrated with training pipeline

### 5. Character Chatbots
- **LLama-based chatbot**: Enhanced with custom training
- **Qwen-based chatbot**: Alternative model architecture
- Both support dynamic model loading from S3

## S3 Structure

Your S3 bucket will have this structure:

```
your-bucket/
├── models/
│   └── trained/
│       ├── llama/
│       │   ├── latest.json
│       │   └── 20241224-143022/  # Timestamp folders
│       └── qwen/
│           ├── latest.json
│           └── 20241224-143022/
└── data/
    ├── training/
    │   └── Subtitles/
    ├── processed/
    └── training_outputs/
```

### Model Info JSON Format

Each `latest.json` contains:
```json
{
  "model_path": "models/trained/llama/20241224-143022/",
  "timestamp": "20241224-143022",
  "accuracy": 0.85,
  "loss": 0.45,
  "training_data": "Stranger Things subtitles",
  "model_type": "llama",
  "description": "Trained model description"
}
```

## Deployment Options

### Local Development
```bash
python gradio_app_v2.py
```

### EC2 Deployment
1. Launch EC2 instance with Python 3.8+
2. Install dependencies
3. Configure AWS credentials
4. Clone your code
5. Run the deployment script
6. Start the app

### Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "gradio_app_v2.py"]
```

## Troubleshooting

### Common Issues

1. **AWS Credentials Error**: Make sure AWS CLI is configured or environment variables are set
2. **Missing Models**: App will use HuggingFace fallbacks if S3 models aren't found
3. **Import Errors**: Run the deployment script to check all dependencies
4. **Port Issues**: Change `server_port` in `config.py` if 7860 is busy

### Debug Mode

The app runs with `debug=True` by default. Check the terminal output for detailed error messages.

### Model Loading Issues

If models fail to load from S3:
1. Check S3 bucket permissions
2. Verify `latest.json` files exist
3. Use "Refresh Models from S3" button in the app
4. Check AWS CloudWatch logs for detailed errors

## Advanced Usage

### Custom Training Data

To use your own training data:
1. Upload subtitle files to S3: `data/training/Subtitles/`
2. Run the training pipeline
3. Models will be automatically uploaded and made available

### Multiple Model Versions

The system supports multiple model versions:
- Each training run creates a timestamped folder
- `latest.json` points to the most recent model
- You can manually update `latest.json` to use specific versions

### Model Comparison

Compare different model versions by:
1. Training multiple models with different parameters
2. Updating `latest.json` to switch between versions
3. Using the "Refresh Models" feature to reload

## Performance Optimization

### For Large Models
- Use GPU-enabled EC2 instances
- Increase instance memory
- Enable model quantization in your training code

### For Production
- Use Application Load Balancer
- Set up auto-scaling
- Enable CloudWatch monitoring
- Use S3 Transfer Acceleration for faster model loading

## Security Considerations

- Keep your HuggingFace tokens secure
- Use IAM roles instead of access keys when possible
- Enable S3 bucket versioning for model backup
- Set appropriate S3 bucket policies

## Contributing

When adding new features:
1. Update `config.py` for new settings
2. Add corresponding functions to the training pipeline
3. Update the Gradio interface in `gradio_app_v2.py`
4. Test with the deployment script

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review AWS CloudWatch logs
3. Enable debug mode for detailed error messages
4. Check S3 bucket structure and permissions