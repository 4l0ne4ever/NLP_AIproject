# Stranger Things NLP - Character Chatbot Project

**Interactive Character Chatbots from Netflix's Stranger Things**

**Link to trained models:** https://huggingface.co/christopherxzyx

**[Complete Project Guide](PROJECT_GUIDE.md)** - Comprehensive documentation for setup, development, and deployment

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

### AWS Production Deployment
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

**For detailed instructions, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md)**

---

## Project Structure

```
project/
├── PROJECT_GUIDE.md        # Complete documentation
├── README.md              # This file
├── WARP.md               # Development guidelines
├── gradio_app.py         # Main web interface
├── deploy_aws.py         # AWS deployment CLI
├── aws_config.yaml       # AWS configuration
├── requirements.txt      # Dependencies
├── venv/                # Python environment
├── aws/                 # AWS infrastructure code
├── character_chatbot/   # Chatbot models
├── theme_classifier/    # Theme analysis
├── character_network/   # Network analysis
├── text_classification/ # Location classifier
├── utils/              # Utilities
├── data/               # Training data
└── stubs/              # Output results
```
