# Directory Tree - SageMaker Version

## Overview
This document provides a comprehensive directory structure for the SageMaker deployment version of the Stranger Things NLP project, highlighting the cloud-native, managed services approach and the role of each file in the SageMaker ecosystem.

## Complete Directory Structure

```
stranger-things-nlp/                          # Project root directory
├── README.md                                 # Main project documentation
├── requirements.txt                          # Python dependencies (includes SageMaker SDK)
├── WARP.md                                   # Development guide for WARP terminal
├── PROJECT_GUIDE.md                         # Comprehensive project guide
├── ARCHITECTURE_SAGEMAKER.md                # SageMaker architecture documentation
├── SAGEMAKER_INTEGRATION.md                 # Detailed SageMaker integration guide
├── .env                                      # Environment variables (HuggingFace tokens, AWS config)
├── .gitignore                               # Git ignore rules (includes SageMaker artifacts)
│
├── sagemaker/                               # SageMaker Integration Package
│   ├── __init__.py                          # Package initialization
│   ├── README.md                            # SageMaker module documentation
│   ├── sagemaker_config.yaml               # SageMaker-specific configuration
│   │                                        # - Training job settings
│   │                                        # - Endpoint configurations
│   │                                        # - S3 bucket mappings
│   │                                        # - IAM role specifications
│   ├── config.py                           # SageMaker configuration management
│   │                                        # - Training configuration classes
│   │                                        # - Endpoint configuration classes
│   │                                        # - S3 path management
│   │                                        # - Resource naming conventions
│   ├── training_orchestrator.py            # Training job orchestration
│   │                                        # - SageMaker training job lifecycle
│   │                                        # - Custom training container management
│   │                                        # - Hyperparameter tuning jobs
│   │                                        # - Model registry integration
│   │                                        # - Spot instance management
│   ├── deployment_manager.py               # Model deployment and endpoint management
│   │                                        # - Real-time endpoint deployment
│   │                                        # - Batch transform job management
│   │                                        # - Multi-model endpoint support
│   │                                        # - Auto-scaling configuration
│   │                                        # - A/B testing capabilities
│   ├── storage.py                          # SageMaker-optimized S3 operations
│   │                                        # - Training data upload/download
│   │                                        # - Model artifact management
│   │                                        # - SageMaker-specific S3 paths
│   │                                        # - Data preprocessing pipelines
│   ├── monitoring.py                       # CloudWatch and SageMaker monitoring
│   │                                        # - Custom metric creation
│   │                                        # - Training job monitoring
│   │                                        # - Endpoint performance tracking
│   │                                        # - Cost analysis and alerts
│   ├── gradio_app.py                       # SageMaker-enabled web interface
│   │                                        # - SageMaker endpoint integration
│   │                                        # - Real-time model switching
│   │                                        # - Batch processing interface
│   │                                        # - Performance monitoring dashboard
│   ├── deploy.py                           # Unified SageMaker deployment CLI
│   │                                        # - Infrastructure setup automation
│   │                                        # - Training job management
│   │                                        # - Endpoint deployment
│   │                                        # - Resource cleanup utilities
│   ├── test_train.py                       # SageMaker training validation
│   │                                        # - Training job testing
│   │                                        # - Model validation
│   │                                        # - Performance benchmarking
│   │
│   ├── docker/                             # Custom Docker containers for SageMaker
│   │   ├── Dockerfile                      # Training container definition
│   │   │                                   # - PyTorch + HuggingFace environment
│   │   │                                   # - CUDA support for GPU training
│   │   │                                   # - Custom dependencies
│   │   ├── inference/                      # Inference container configurations
│   │   │   ├── Dockerfile                 # Inference-optimized container
│   │   │   ├── predictor.py              # Custom inference logic
│   │   │   └── requirements.txt          # Inference dependencies
│   │   ├── build_and_push.sh             # Container build and ECR push script
│   │   └── train.py                       # Training entry point script
│   │                                      # - SageMaker training job entry
│   │                                      # - Hyperparameter parsing
│   │                                      # - Model saving to S3
│   │
│   ├── notebooks/                          # Jupyter notebooks for SageMaker Studio
│   │   ├── training_experiments.ipynb     # Interactive training experiments
│   │   ├── model_evaluation.ipynb         # Model performance analysis
│   │   ├── hyperparameter_tuning.ipynb   # HPO job configuration
│   │   └── deployment_testing.ipynb      # Endpoint testing and validation
│   │
│   └── examples/                           # SageMaker usage examples
│       ├── basic_training.py              # Simple training job example
│       ├── distributed_training.py       # Multi-instance training
│       ├── batch_processing.py           # Batch transform examples
│       └── endpoint_testing.py           # Endpoint testing utilities
│
├── character_chatbot/                      # Character Chatbot Components (SageMaker-compatible)
│   ├── __init__.py                         # Package initialization
│   ├── character_chatbot.py               # SageMaker-compatible Llama chatbot
│   │                                       # - SageMaker training integration
│   │                                       # - Model registry publishing
│   │                                       # - Endpoint-compatible inference
│   ├── character_chatbotQwen.py           # SageMaker-compatible Qwen chatbot
│   │                                       # - Custom container training
│   │                                       # - SageMaker model packaging
│   ├── sagemaker_trainer.py               # SageMaker-specific training logic
│   │                                       # - Training job orchestration
│   │                                       # - Hyperparameter management
│   │                                       # - Model artifact organization
│   ├── evaluate.py                        # Model evaluation (SageMaker integrated)
│   │                                       # - Endpoint-based evaluation
│   │                                       # - Batch evaluation jobs
│   │                                       # - A/B testing metrics
│   ├── bleuScore.py                       # BLEU score calculation
│   │                                       # - SageMaker processing job support
│   │                                       # - Batch metric calculation
│   └── inference.py                       # SageMaker inference handler
│                                          # - Model loading for endpoints
│                                          # - Input/output transformation
│                                          # - Error handling and logging
│
├── text_classification/                    # Text Classification (SageMaker-optimized)
│   ├── __init__.py                         # Package initialization
│   ├── location_classifier.py             # SageMaker-compatible classifier
│   │                                       # - SageMaker training pipeline
│   │                                       # - Endpoint deployment support
│   ├── sagemaker_trainer.py               # SageMaker training orchestration
│   │                                       # - Custom training containers
│   │                                       # - Distributed training support
│   ├── custom_trainer.py                  # Enhanced HuggingFace trainer
│   │                                       # - SageMaker metrics integration
│   │                                       # - Checkpointing for fault tolerance
│   ├── training_utils.py                  # SageMaker training utilities
│   │                                       # - Data preprocessing for SageMaker
│   │                                       # - Model packaging utilities
│   └── inference.py                       # SageMaker inference handler
│                                          # - Multi-model endpoint support
│                                          # - Batch transform compatibility
│
├── theme_classifier/                       # Theme Classification (SageMaker-enabled)
│   ├── __init__.py                         # Package initialization
│   ├── theme_classifier.py                # SageMaker-compatible theme classifier
│   │                                       # - Serverless inference support
│   │                                       # - Batch processing optimization
│   └── inference.py                       # SageMaker inference handler
│                                          # - Zero-shot classification endpoints
│                                          # - Batch transform support
│
├── character_network/                      # Character Network Analysis (SageMaker-optimized)
│   ├── __init__.py                         # Package initialization
│   ├── named_entity_recognizer.py         # SageMaker-compatible NER
│   │                                       # - Processing job integration
│   │                                       # - Batch entity extraction
│   ├── character_network_generator.py     # Network generation (SageMaker)
│   │                                       # - SageMaker processing jobs
│   │                                       # - S3-based result storage
│   └── batch_processor.py                 # SageMaker batch processing
│                                          # - Large-scale network analysis
│                                          # - Distributed graph computation
│
├── utils/                                  # Shared Utilities (SageMaker-optimized)
│   ├── __init__.py                         # Package initialization
│   ├── data_loader.py                     # SageMaker data loading utilities
│   │                                       # - S3-optimized data loading
│   │                                       # - SageMaker input modes
│   │                                       # - Data validation for training
│   └── sagemaker_utils.py                 # SageMaker helper functions
│                                          # - ARN parsing and validation
│                                          # - Resource naming conventions
│                                          # - Common SageMaker operations
│
├── gradio_app.py                          # Legacy Gradio app (for local testing)
│                                          # - Local model testing
│                                          # - Development interface
│                                          # - Compatibility with SageMaker version
│
├── deploy_aws.py                          # Legacy AWS deployment (EC2)
│                                          # - Maintained for migration reference
│                                          # - Comparison with SageMaker approach
│
├── infrastructure/                        # Infrastructure as Code
│   ├── cloudformation/                    # CloudFormation templates
│   │   ├── sagemaker_resources.yaml      # SageMaker infrastructure
│   │   ├── iam_roles.yaml               # IAM roles and policies
│   │   └── s3_buckets.yaml              # S3 bucket configurations
│   ├── terraform/                         # Terraform configurations (optional)
│   │   ├── main.tf                       # Main Terraform configuration
│   │   ├── variables.tf                  # Variable definitions
│   │   └── outputs.tf                    # Output definitions
│   └── cdk/                              # AWS CDK stacks (optional)
│       ├── app.py                        # CDK application entry point
│       └── sagemaker_stack.py           # SageMaker resources stack
│
├── data/                                  # Data management (SageMaker-optimized)
│   ├── raw/                               # Raw data (uploaded to S3)
│   │   ├── subtitles/                    # Original subtitle files
│   │   ├── transcripts/                  # Processed transcripts
│   │   └── characters/                   # Character-specific datasets
│   ├── processed/                         # Processed data (SageMaker format)
│   │   ├── training/                     # SageMaker training format
│   │   ├── validation/                   # Validation datasets
│   │   └── inference/                    # Inference test data
│   └── s3_mappings.yaml                  # S3 path configurations
│                                         # - Training data locations
│                                         # - Model artifact paths
│                                         # - Result storage paths
│
├── models/                               # Model configurations and metadata
│   ├── chatbot/                          # Chatbot model configurations
│   │   ├── llama_config.yaml            # Llama model settings
│   │   ├── qwen_config.yaml             # Qwen model settings
│   │   └── training_params.yaml         # Training hyperparameters
│   ├── classification/                    # Classification model configs
│   │   ├── location_config.yaml         # Location classifier settings
│   │   └── training_params.yaml         # Training hyperparameters
│   └── model_registry.yaml              # Model registry metadata
│                                         # - Model versions
│                                         # - Approval status
│                                         # - Performance metrics
│
├── scripts/                              # Automation scripts (SageMaker-focused)
│   ├── setup_sagemaker.sh               # SageMaker environment setup
│   ├── build_containers.sh              # Docker container build automation
│   ├── deploy_infrastructure.sh          # Infrastructure deployment
│   ├── cleanup_resources.sh             # Resource cleanup and cost optimization
│   └── migrate_from_ec2.sh              # Migration utilities from EC2
│
├── config/                               # Configuration files
│   ├── sagemaker_config.yaml            # Main SageMaker configuration
│   ├── training_configs/                 # Training-specific configurations
│   │   ├── chatbot_training.yaml        # Chatbot training settings
│   │   └── classification_training.yaml # Classification training settings
│   ├── deployment_configs/               # Deployment configurations
│   │   ├── endpoint_configs.yaml        # Endpoint configurations
│   │   └── batch_configs.yaml           # Batch processing configs
│   └── monitoring_configs/               # Monitoring and alerting
│       ├── cloudwatch_dashboards.yaml   # Dashboard configurations
│       └── alerts.yaml                  # Alert configurations
│
├── tests/                                # Test suite (SageMaker-focused)
│   ├── unit/                             # Unit tests
│   │   ├── test_sagemaker_config.py     # Configuration testing
│   │   ├── test_training_orchestrator.py # Training orchestration tests
│   │   ├── test_deployment_manager.py    # Deployment testing
│   │   └── test_monitoring.py           # Monitoring functionality tests
│   ├── integration/                      # Integration tests
│   │   ├── test_training_pipeline.py    # End-to-end training tests
│   │   ├── test_endpoint_deployment.py  # Endpoint deployment tests
│   │   ├── test_batch_processing.py     # Batch processing tests
│   │   └── test_model_registry.py       # Model registry tests
│   ├── performance/                      # Performance tests
│   │   ├── test_endpoint_latency.py     # Latency benchmarking
│   │   ├── test_throughput.py           # Throughput testing
│   │   └── test_cost_analysis.py        # Cost optimization tests
│   └── fixtures/                         # Test data and fixtures
│       ├── sample_training_data/         # Sample datasets
│       ├── mock_responses/               # Mock API responses
│       └── test_models/                  # Small test models
│
├── monitoring/                           # Monitoring and observability
│   ├── dashboards/                       # CloudWatch dashboards
│   │   ├── training_dashboard.json      # Training metrics dashboard
│   │   ├── endpoint_dashboard.json      # Endpoint performance dashboard
│   │   └── cost_dashboard.json          # Cost analysis dashboard
│   ├── alerts/                           # CloudWatch alerts
│   │   ├── training_alerts.yaml         # Training job alerts
│   │   ├── endpoint_alerts.yaml         # Endpoint performance alerts
│   │   └── cost_alerts.yaml             # Cost threshold alerts
│   └── logs/                             # Log analysis tools
│       ├── log_parser.py                # CloudWatch log parsing
│       └── metrics_analyzer.py          # Custom metrics analysis
│
├── docs/                                 # Documentation
│   ├── sagemaker/                        # SageMaker-specific documentation
│   │   ├── training_guide.md            # Training job guide
│   │   ├── deployment_guide.md          # Deployment guide
│   │   ├── monitoring_guide.md          # Monitoring and alerting
│   │   └── cost_optimization.md         # Cost optimization strategies
│   ├── api/                              # API documentation
│   │   ├── training_api.md              # Training orchestrator API
│   │   └── deployment_api.md            # Deployment manager API
│   ├── architecture/                     # Architecture documentation
│   │   ├── sagemaker_architecture.md    # Detailed architecture
│   │   └── data_flow_diagrams/          # Data flow visualizations
│   └── tutorials/                        # Step-by-step tutorials
│       ├── getting_started.md           # Quick start guide
│       ├── advanced_training.md         # Advanced training scenarios
│       └── production_deployment.md     # Production best practices
│
└── .github/                              # GitHub workflows (CI/CD)
    └── workflows/                        # Automated workflows
        ├── sagemaker_test.yml           # SageMaker testing workflow
        ├── container_build.yml          # Docker container build
        ├── deploy_staging.yml           # Staging deployment
        ├── deploy_production.yml        # Production deployment
        └── cost_monitoring.yml          # Automated cost reporting
```

## Key File Responsibilities by Category

### 1. **SageMaker Integration Core** (`sagemaker/`)

#### Configuration and Orchestration
- **config.py**: Centralized SageMaker configuration management with classes for training, endpoints, and monitoring
- **training_orchestrator.py**: Complete training job lifecycle management including HPO and model registry
- **deployment_manager.py**: Endpoint deployment, scaling, and management with A/B testing capabilities
- **storage.py**: S3 operations optimized for SageMaker workflows and data formats

#### Application Integration
- **gradio_app.py**: SageMaker-enabled web interface with real-time endpoint integration
- **deploy.py**: Unified CLI for all SageMaker operations from setup to cleanup
- **monitoring.py**: CloudWatch integration with custom metrics and automated alerting

#### Container Management
- **docker/**: Custom training and inference containers optimized for SageMaker
- **docker/train.py**: SageMaker training entry point with hyperparameter parsing
- **docker/inference/predictor.py**: Custom inference logic for endpoints

### 2. **SageMaker-Compatible NLP Components**

#### Enhanced Chatbot Components (`character_chatbot/`)
- **sagemaker_trainer.py**: SageMaker-specific training orchestration with model registry publishing
- **inference.py**: Endpoint-compatible inference handlers with input/output transformation

#### Cloud-Native Classification (`text_classification/`)
- **sagemaker_trainer.py**: Distributed training support with SageMaker containers
- **inference.py**: Multi-model endpoint support for classification tasks

#### Batch-Optimized Analysis (`character_network/`)
- **batch_processor.py**: Large-scale network analysis using SageMaker processing jobs

### 3. **Infrastructure as Code** (`infrastructure/`)
- **cloudformation/**: Complete AWS infrastructure templates for reproducible deployments
- **terraform/**: Alternative infrastructure provisioning (optional)
- **cdk/**: AWS CDK stacks for programmatic infrastructure management

### 4. **Advanced Monitoring and Observability** (`monitoring/`)
- **dashboards/**: Pre-configured CloudWatch dashboards for training and inference monitoring
- **alerts/**: Comprehensive alerting for performance, cost, and operational issues
- **logs/**: Log analysis and metrics extraction tools

### 5. **Production-Ready Testing** (`tests/`)
- **performance/**: Dedicated performance and cost optimization testing
- **integration/**: End-to-end SageMaker workflow testing
- **fixtures/**: Comprehensive test data and mock objects

## SageMaker-Specific Features

### 1. **Managed Training Infrastructure**
- **Automatic Scaling**: Training jobs automatically scale based on workload
- **Spot Instance Management**: Automatic spot instance handling with checkpointing
- **Hyperparameter Optimization**: Built-in HPO with Bayesian optimization
- **Model Registry**: Automated model versioning and approval workflows

### 2. **Production Deployment Capabilities**
- **Real-time Endpoints**: Auto-scaling inference endpoints with load balancing
- **Batch Processing**: Cost-effective batch transform jobs for large datasets
- **Multi-Model Endpoints**: Efficient resource sharing across multiple models
- **A/B Testing**: Built-in traffic splitting for model comparison

### 3. **Enterprise Monitoring and Governance**
- **Built-in Monitoring**: CloudWatch integration with custom metrics
- **Cost Optimization**: Automated cost tracking and optimization recommendations
- **Security**: VPC integration, encryption, and IAM-based access control
- **Compliance**: Enterprise-grade compliance and audit capabilities

### 4. **Cloud-Native Development**
- **SageMaker Studio**: Integrated development environment for ML workflows
- **Jupyter Notebooks**: Interactive development and experimentation
- **Container Management**: Custom container support via ECR integration
- **Pipeline Orchestration**: SageMaker Pipelines for MLOps workflows

## Migration Benefits from EC2

### 1. **Operational Efficiency**
- **70-80% reduction** in infrastructure management overhead
- **Automatic scaling** eliminates manual capacity planning
- **Built-in monitoring** reduces custom monitoring setup
- **Managed updates** eliminate OS and dependency management

### 2. **Cost Optimization**
- **20-40% cost reduction** through efficient resource utilization
- **Spot instance support** with automatic interruption handling
- **Pay-per-use model** for inference workloads
- **Automated resource cleanup** prevents cost leakage

### 3. **Enhanced Reliability**
- **Built-in fault tolerance** with automatic recovery
- **Multi-AZ deployment** for high availability
- **Automated backup** and disaster recovery
- **Service-level agreements** for uptime guarantees

### 4. **Advanced ML Capabilities**
- **Distributed training** support out of the box
- **Model versioning** and approval workflows
- **A/B testing** capabilities for model comparison
- **Advanced monitoring** with data quality and model drift detection

## Usage Patterns

### Development Workflow
1. **Local Development**: Use SageMaker Studio or local notebooks for experimentation
2. **Container Development**: Build and test custom containers locally
3. **Training Jobs**: Launch managed training jobs with automatic scaling
4. **Model Registry**: Automatic model versioning and metadata tracking
5. **Endpoint Deployment**: Deploy models to managed inference endpoints
6. **Monitoring**: Real-time monitoring and automated alerting

### Production Workflow
1. **Infrastructure Setup**: Deploy via CloudFormation or Terraform templates
2. **CI/CD Integration**: Automated testing and deployment pipelines
3. **Model Training**: Scheduled or event-driven training jobs
4. **Model Approval**: Automated or manual model approval workflows
5. **Blue/Green Deployment**: Zero-downtime model updates
6. **Continuous Monitoring**: Performance, cost, and compliance monitoring

This SageMaker-optimized directory structure provides a comprehensive foundation for production-scale machine learning operations while maintaining the flexibility needed for experimentation and development.