# Directory Tree - EC2/S3 Version

## Overview
This document provides a comprehensive directory structure for the EC2/S3 deployment version of the Stranger Things NLP project, detailing the role and purpose of each file and directory.

## Complete Directory Structure

```
stranger-things-nlp/                          # Project root directory
├── README.md                                 # Main project documentation
├── requirements.txt                          # Python dependencies for local/EC2 deployment
├── WARP.md                                   # Development guide for WARP terminal
├── PROJECT_GUIDE.md                         # Comprehensive project guide
├── ARCHITECTURE_EC2_S3.md                   # EC2/S3 architecture documentation
├── aws_config.yaml                          # AWS infrastructure configuration
├── .env                                      # Environment variables (not in git)
├── .gitignore                               # Git ignore rules
│
├── aws/                                      # AWS Infrastructure Management
│   ├── __init__.py                          # Package initialization
│   ├── config.py                            # AWS configuration management
│   │                                        # - EC2 instance settings
│   │                                        # - S3 bucket configurations
│   │                                        # - Security group definitions
│   │                                        # - IAM role specifications
│   ├── ec2_manager.py                       # EC2 instance lifecycle management
│   │                                        # - Launch training instances
│   │                                        # - Launch Gradio hosting instances
│   │                                        # - Instance monitoring and termination
│   │                                        # - SSH key and security group management
│   ├── ec2_orchestrator.py                 # High-level orchestration workflows
│   │                                        # - End-to-end training orchestration
│   │                                        # - Deployment automation
│   │                                        # - Multi-instance coordination
│   └── storage.py                           # S3 operations and data management
│                                           # - Model artifact upload/download
│                                           # - Training data synchronization
│                                           # - Result storage and retrieval
│
├── character_chatbot/                       # Character Chatbot Components
│   ├── __init__.py                          # Package initialization
│   ├── character_chatbot.py                # Main Llama-based chatbot class
│   │                                        # - Model loading and initialization
│   │                                        # - Training pipeline implementation
│   │                                        # - Inference and chat functionality
│   │                                        # - S3 integration for model storage
│   ├── character_chatbotQwen.py            # Alternative Qwen-based implementation
│   │                                        # - Qwen model fine-tuning
│   │                                        # - Character-specific response generation
│   │                                        # - Memory management for conversations
│   ├── character_chatbot_s3.py             # S3-enabled chatbot variant
│   │                                        # - Direct S3 model loading
│   │                                        # - Cloud-native training pipeline
│   │                                        # - Distributed training support
│   ├── evaluate.py                         # Model evaluation utilities
│   │                                        # - Performance metrics calculation
│   │                                        # - Validation set evaluation
│   │                                        # - A/B testing support
│   ├── bleuScore.py                        # BLEU score calculation
│   │                                        # - Translation quality metrics
│   │                                        # - Response quality evaluation
│   │                                        # - Benchmark comparisons
│   └── review.py                           # Model review and analysis
│                                           # - Training progress visualization
│                                           # - Loss curve analysis
│                                           # - Model comparison utilities
│
├── text_classification/                     # Text Classification Components
│   ├── __init__.py                          # Package initialization
│   ├── location_classifier.py              # Location classification model
│   │                                        # - DistilBERT fine-tuning
│   │                                        # - Location entity recognition
│   │                                        # - Inference pipeline
│   ├── custom_trainer.py                   # Custom HuggingFace trainer
│   │                                        # - Class-weighted loss functions
│   │                                        # - Custom metrics implementation
│   │                                        # - Advanced training strategies
│   └── training_utils.py                   # Training utility functions
│                                           # - Data preprocessing pipelines
│                                           # - Evaluation metrics
│                                           # - Model serialization helpers
│
├── theme_classifier/                        # Theme Classification Components
│   ├── __init__.py                          # Package initialization
│   └── theme_classifier.py                # Zero-shot theme classification
│                                           # - DistilBART-based classification
│                                           # - Custom theme detection
│                                           # - Batch processing capabilities
│
├── character_network/                       # Character Network Analysis
│   ├── __init__.py                          # Package initialization
│   ├── named_entity_recognizer.py         # NER implementation
│   │                                        # - SpaCy-based entity extraction
│   │                                        # - Character name recognition
│   │                                        # - Entity relationship detection
│   └── character_network_generator.py     # Network graph generation
│                                           # - PyVis interactive visualizations
│                                           # - Network analysis algorithms
│                                           # - Graph export and storage
│
├── utils/                                   # Shared Utilities
│   ├── __init__.py                          # Package initialization
│   └── data_loader.py                      # Data loading and preprocessing
│                                           # - Subtitle file parsing (.srt)
│                                           # - DataFrame creation and management
│                                           # - Data validation and cleaning
│
├── crawler/                                # Data Collection Tools
│   ├── characters.py                       # Character data extraction
│   │                                        # - Character dialogue collection
│   │                                        # - Character metadata extraction
│   │                                        # - Data format standardization
│   └── locations.py                        # Location data extraction
│                                           # - Location reference collection
│                                           # - Geographic data processing
│                                           # - Location classification datasets
│
├── gradio_app.py                           # Main web application
│                                           # - Multi-tab Gradio interface
│                                           # - Theme classification UI
│                                           # - Character network visualization
│                                           # - Text classification interface
│                                           # - Character chatbot integration
│
├── deploy_aws.py                           # AWS deployment orchestration
│                                           # - Infrastructure deployment
│                                           # - Training job coordination
│                                           # - Model deployment automation
│                                           # - Resource cleanup utilities
│
├── clean_data.py                           # Data preprocessing utility
│                                           # - Dataset cleaning operations
│                                           # - Format standardization
│                                           # - Quality assurance checks
│
├── get_char_list.py                        # Character list extraction
│                                           # - Character name extraction
│                                           # - Character frequency analysis
│                                           # - Character filtering utilities
│
├── test_deployment.py                      # Deployment testing
│                                           # - End-to-end deployment tests
│                                           # - Integration testing
│                                           # - Performance validation
│
├── convert_key.py                          # SSH key format converter
│                                           # - RSA to OpenSSH format conversion
│                                           # - Key validation utilities
│                                           # - Paramiko compatibility fixes
│
├── data/                                   # Local data storage (development)
│   ├── raw/                                # Raw input data
│   │   ├── subtitles/                      # Original subtitle files (.srt)
│   │   ├── transcripts/                    # Processed transcripts
│   │   └── characters/                     # Character-specific datasets
│   ├── processed/                          # Processed datasets
│   │   ├── theme_classification/           # Theme analysis results
│   │   ├── network_analysis/               # Character network data
│   │   └── text_classification/            # Classification datasets
│   └── models/                            # Local model storage (development)
│       ├── chatbots/                       # Character chatbot models
│       ├── classifiers/                    # Text classification models
│       └── embeddings/                     # Pre-trained embeddings
│
├── logs/                                   # Local log storage (development)
│   ├── training/                           # Training logs
│   ├── deployment/                         # Deployment logs
│   └── application/                        # Application logs
│
├── scripts/                                # Utility and automation scripts
│   ├── setup_ec2.sh                       # EC2 instance setup script
│   ├── install_dependencies.sh            # Dependency installation
│   ├── download_models.sh                 # Model download automation
│   └── backup_data.sh                     # Data backup utilities
│
├── config/                                 # Configuration files
│   ├── aws_config.yaml                    # AWS-specific configurations
│   ├── model_config.yaml                 # Model hyperparameters
│   ├── training_config.yaml              # Training configurations
│   └── deployment_config.yaml            # Deployment settings
│
├── docs/                                  # Documentation
│   ├── api/                               # API documentation
│   ├── architecture/                      # Architecture diagrams
│   ├── deployment/                        # Deployment guides
│   └── user_guide/                        # User manuals
│
├── tests/                                 # Test suite
│   ├── unit/                              # Unit tests
│   │   ├── test_chatbot.py               # Chatbot functionality tests
│   │   ├── test_classification.py        # Classification tests
│   │   └── test_aws_integration.py       # AWS integration tests
│   ├── integration/                       # Integration tests
│   │   ├── test_ec2_deployment.py        # EC2 deployment tests
│   │   └── test_end_to_end.py            # End-to-end workflow tests
│   └── fixtures/                          # Test data and fixtures
│
└── .github/                               # GitHub workflows (if using GitHub)
    └── workflows/                         # CI/CD workflows
        ├── test.yml                       # Automated testing
        └── deploy.yml                     # Automated deployment
```

## Key File Responsibilities by Category

### 1. **Infrastructure Management** (`aws/`)
- **config.py**: Centralized configuration for all AWS resources, including EC2 instance types, S3 bucket names, security groups, and IAM roles
- **ec2_manager.py**: Core EC2 operations including instance launching, monitoring, SSH connections, and termination
- **ec2_orchestrator.py**: High-level workflows that coordinate multiple AWS services for training and deployment
- **storage.py**: S3 integration for data upload/download, model artifact management, and result storage

### 2. **Core NLP Components**

#### Character Chatbots (`character_chatbot/`)
- **character_chatbot.py**: Primary implementation using Llama models with LoRA fine-tuning
- **character_chatbotQwen.py**: Alternative implementation using Qwen models
- **character_chatbot_s3.py**: Cloud-native version with direct S3 integration
- **evaluate.py**: Model evaluation and performance measurement
- **bleuScore.py**: Response quality evaluation using BLEU scores

#### Text Classification (`text_classification/`)
- **location_classifier.py**: DistilBERT-based location classification
- **custom_trainer.py**: Custom HuggingFace trainer with class weighting
- **training_utils.py**: Shared training utilities and helper functions

#### Theme Analysis (`theme_classifier/`)
- **theme_classifier.py**: Zero-shot theme classification using DistilBART

#### Network Analysis (`character_network/`)
- **named_entity_recognizer.py**: Character entity recognition and extraction
- **character_network_generator.py**: Interactive network visualization

### 3. **Application Layer**
- **gradio_app.py**: Unified web interface integrating all NLP components
- **deploy_aws.py**: Main deployment orchestration script
- **test_deployment.py**: Deployment validation and testing

### 4. **Data Management**
- **utils/data_loader.py**: Subtitle parsing and data preprocessing
- **crawler/**: Data collection and extraction tools
- **clean_data.py**: Data cleaning and quality assurance
- **get_char_list.py**: Character extraction and analysis

### 5. **Configuration and Setup**
- **aws_config.yaml**: Infrastructure configuration settings
- **requirements.txt**: Python dependencies for EC2 deployment
- **convert_key.py**: SSH key format conversion for EC2 access

### 6. **Development and Testing**
- **tests/**: Comprehensive test suite for all components
- **scripts/**: Automation scripts for setup and deployment
- **docs/**: Documentation and guides
- **logs/**: Development and debugging logs

## Deployment-Specific Features

### EC2-Optimized Components
1. **Direct SSH Integration**: Files like `ec2_manager.py` handle SSH connections for code deployment
2. **Instance Management**: Comprehensive EC2 lifecycle management including spot instance support
3. **Manual Scaling**: Scripts for manual scaling and resource management
4. **Local Development Support**: Extensive local testing and development capabilities

### S3 Integration
1. **Model Artifact Storage**: Automated upload/download of trained models
2. **Data Synchronization**: Efficient data transfer between local and cloud environments
3. **Result Storage**: Centralized storage for training results and outputs
4. **Backup and Recovery**: Automated backup strategies for critical data

### Security and Access
1. **SSH Key Management**: Automated SSH key setup and conversion
2. **Security Groups**: Programmatic security group creation and management
3. **IAM Integration**: Service roles and policies for EC2-S3 access
4. **Network Configuration**: VPC and subnet management for secure deployments

## Usage Patterns

### Development Workflow
1. **Local Development**: Use local data and models for development
2. **Data Preparation**: Process and upload data to S3
3. **EC2 Deployment**: Launch training instances and deploy code
4. **Model Training**: Execute training jobs on GPU instances
5. **Result Retrieval**: Download results and models from S3
6. **Gradio Deployment**: Host web interface on separate EC2 instance

### Operational Workflow
1. **Infrastructure Setup**: Configure AWS resources via `deploy_aws.py`
2. **Training Pipeline**: Orchestrate training jobs across multiple models
3. **Model Management**: Store and version models in S3
4. **Application Deployment**: Deploy Gradio interface for end users
5. **Monitoring**: Track performance and costs via CloudWatch
6. **Maintenance**: Regular updates and security patches

This directory structure provides a clear separation of concerns while maintaining the flexibility needed for EC2-based deployments, offering both local development capabilities and cloud-scale production deployment options.