# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a comprehensive NLP project that creates character chatbots from the Netflix series Stranger Things. The system combines multiple machine learning approaches including named entity recognition, text classification, network analysis, and large language model fine-tuning to create an interactive character chatbot experience.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variable for HuggingFace token
export huggingface_token="your_token_here"
# or create a .env file with: huggingface_token=your_token_here
```

### Running the Application
```bash
# Launch the main Gradio application
python gradio_app.py
```

### Running Individual Components
```bash
# Run theme classification
python -c "from theme_classifier import ThemeClassifier; tc = ThemeClassifier(['theme1', 'theme2'])"

# Run character network analysis  
python -c "from character_network import NamedEntityRecognizer; ner = NamedEntityRecognizer()"

# Run text classification training/inference
python -c "from text_classification import LocationClassifier; lc = LocationClassifier('model_path', 'data_path')"
```

### Testing and Evaluation
```bash
# Evaluate chatbot models (if evaluation scripts exist)
python character_chatbot/evaluate.py

# Calculate BLEU scores
python character_chatbot/bleuScore.py
```

## High-Level Architecture

### Core Components Architecture

The project follows a modular architecture with four main functional areas:

1. **Data Processing Pipeline**: `utils/data_loader.py` handles subtitle/script parsing from .srt files into structured DataFrames with season/episode metadata.

2. **NLP Analysis Layer**: 
   - **Theme Classifier** (`theme_classifier/`): Uses zero-shot classification with DistilBART to categorize content themes
   - **Character Network** (`character_network/`): Implements NER with SpaCy and generates interactive network graphs using PyVis
   - **Text Classification** (`text_classification/`): Fine-tunes DistilBERT for location classification with custom trainer and class weighting

3. **LLM Integration Layer** (`character_chatbot/`):
   - **Llama-based Chatbot**: Fine-tunes Llama 3.2-3B with LoRA adapters on character-specific dialogue
   - **Qwen-based Chatbot**: Alternative implementation using Qwen models
   - Both use conversational memory and character-specific system prompts

4. **Web Interface** (`gradio_app.py`): Unified Gradio interface that orchestrates all components with separate tabs for each functionality.

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

### Model Training Architecture

- **Theme Classification**: Zero-shot learning eliminates need for labeled training data
- **Location Classification**: Supervised learning with automatic class weight balancing for imbalanced datasets
- **Character Chatbots**: Self-supervised fine-tuning using character dialogue as training pairs with LoRA for efficient parameter updates

## Key Implementation Details

### Model Management
- Models are automatically downloaded from HuggingFace Hub if they exist
- If models don't exist, training is triggered automatically with the provided data
- Both local and remote model storage supported via HuggingFace integration

### Memory Optimization
- Uses 4-bit quantization (BitsAndBytesConfig) for memory-efficient inference
- LoRA adapters minimize memory footprint during fine-tuning
- Explicit garbage collection and GPU cache clearing after training

### Device Compatibility
- Automatic device detection (CUDA → MPS → CPU fallback)
- Models automatically moved to optimal available hardware
- Cross-platform support for different GPU architectures

### Environment Requirements
- Requires `huggingface_token` environment variable for model access and pushing
- Supports both .env files and direct environment variables
- SpaCy model `en_core_web_trf` auto-downloaded during setup

## Data Requirements

- **Subtitle Data**: Place .srt files in organized directories for theme/network analysis
- **Character Data**: CSV files with character dialogue in `/content/data/transcripts/` for chatbot training
- **Location Data**: JSONL files with location descriptions for classification training
- **Expected Data Format**: Subtitle files should follow standard .srt format with season/episode naming (e.g., S01E01.srt)

## Model Endpoints

The project integrates with several HuggingFace models:
- **Theme Classification**: `valhalla/distilbart-mnli-12-3`
- **Text Classification**: `distilbert/distilbert-base-uncased` 
- **Character Chatbots**: 
  - `christopherxzyx/StrangerThings_Llama-3-8B_v3`
  - `christopherxzyx/StrangerThings_Qwen-3-4B`
- **Base Models**: `meta-llama/Llama-3.2-3B-Instruct` for fine-tuning

## Development Notes

### Character Chatbot Training
The chatbot training process automatically:
- Filters dialogue to character-specific responses (currently hardcoded to "Eleven")
- Creates conversational pairs using sliding window approach
- Applies LoRA fine-tuning with gradient checkpointing for memory efficiency
- Merges and pushes final model to HuggingFace Hub

### Network Analysis
Character networks use a sliding window approach (default: 10 sentences) to establish entity relationships, generating interactive visualizations that can be embedded in web interfaces.

### Custom Training Components
- `CustomTrainer` extends HuggingFace Trainer with weighted loss functions
- Automatic class weight calculation for handling imbalanced datasets
- Device-aware training with proper tensor placement
