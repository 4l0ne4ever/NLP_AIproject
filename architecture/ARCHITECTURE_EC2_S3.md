# Stranger Things AI Analysis Suite - EC2/S3 Architecture

Detailed architecture documentation for the EC2/S3 deployment strategy.

## Overview

This architecture uses AWS EC2 for compute resources and S3 for model storage, providing a scalable and cost-effective solution for training and deploying AI models.

## Architecture Diagram

```
                           AWS Cloud
    ┌───────────────────────────────────────────────────────────────────┐
    │                                                                   │
    │    ┌─────────────────────┐      ┌─────────────────────┐    │
    │    │   Training Instances   │      │   Hosting Instance    │    │
    │    │    (g4dn.xlarge)     │      │    (t3.medium)      │    │
    │    │                     │      │                     │    │
    │    │  - LLaMA Training    │      │  - Gradio App        │    │
    │    │  - Qwen Training     │      │  - Model Loading     │    │
    │    │  - Data Processing   │      │  - User Interface    │    │
    │    └─────────────────────┘      └─────────────────────┘    │
    │                │                            │                 │
    │                │                            │                 │
    │                │           S3 Storage         │                 │
    │                │    ┌─────────────────────┐    │                 │
    │                └────│  - Trained Models   │────┘                 │
    │                     │  - Training Data    │                      │
    │                     │  - Model Metadata   │                      │
    │                     │  - Results & Logs   │                      │
    │                     └─────────────────────┘                      │
    │                                                                   │
    └───────────────────────────────────────────────────────────────────┘
                                      │
                                      │
                     External Services
                ┌──────────────────────────────────┐
                │     HuggingFace Hub           │
                │   (Fallback Models)          │
                │                               │
                │  - LLaMA Fallback Model      │
                │  - Qwen Fallback Model       │
                └──────────────────────────────────┘
```

## Components

### EC2 Training Instances
- **Instance Type**: g4dn.xlarge (GPU-enabled)
- **Purpose**: Model training and fine-tuning
- **Features**:
  - NVIDIA T4 GPU for accelerated training
  - 16 GB RAM
  - High-bandwidth network
  - Spot instance support for cost optimization

### EC2 Hosting Instance
- **Instance Type**: t3.medium (CPU-only)
- **Purpose**: Gradio application hosting
- **Features**:
  - 4 GB RAM
  - Moderate CPU performance
  - Cost-effective for web hosting
  - Auto-scaling potential

### S3 Storage
- **Purpose**: Centralized model and data storage
- **Structure**:
  ```
  stranger-things-nlp-bucket/
  ├── models/
  │   └── trained/
  │       ├── llama/
  │       │   ├── latest.json
  │       │   └── 20241224-143022/
  │       └── qwen/
  │           ├── latest.json
  │           └── 20241224-150045/
  ├── checkpoints/
  │   ├── llama/
  │   └── qwen/
  ├── data/
  │   ├── training/
  │   │   └── transcripts/
  │   └── processed/
  └── logs/
  ```

## Deployment Workflow

### 1. Infrastructure Setup
```bash
# Initialize AWS resources
python deploy_aws.py init

# This creates:
# - S3 bucket with proper structure
# - Security groups
# - IAM roles (if needed)
# - Key pair verification
```

### 2. Data Upload
```bash
# Upload training data
python deploy_aws.py upload-data

# Uploads:
# - Transcript CSV files
# - Configuration files
# - Supporting data
```

### 3. Model Training
```bash
# Launch training instances
python deploy_aws.py train llama
python deploy_aws.py train qwen

# Process:
# 1. Launch g4dn.xlarge instance
# 2. Deploy code and dependencies
# 3. Start training pipeline
# 4. Upload results to S3
# 5. Terminate instance
```

### 4. Application Deployment
```bash
# Deploy Gradio application
python deploy_aws.py deploy-gradio

# Process:
# 1. Launch t3.medium instance
# 2. Deploy application code
# 3. Start Gradio server
# 4. Configure model loading
```

## Data Flow

### Training Phase
1. **Data Ingestion**: Transcript files uploaded to S3
2. **Instance Launch**: Training instance started with GPU support
3. **Code Deployment**: Training pipeline deployed to instance
4. **Data Download**: Training data downloaded from S3
5. **Model Training**: Fine-tuning process executed
6. **Result Upload**: Trained models uploaded to S3
7. **Cleanup**: Training instance terminated

### Inference Phase
1. **App Launch**: Gradio app starts on hosting instance
2. **Model Check**: App checks S3 for trained models
3. **Model Download**: Models cached locally if available
4. **Fallback Decision**: Falls back to HuggingFace if needed
5. **User Interaction**: Web interface serves user requests
6. **Model Inference**: Responses generated using loaded models

## Security

### Network Security
- **Security Groups**: Restricted port access (22, 7860, 80, 443)
- **VPC Configuration**: Default VPC with public subnets
- **Key Pair Authentication**: SSH access via key pairs only

### IAM Roles and Policies
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::stranger-things-nlp-bucket",
        "arn:aws:s3:::stranger-things-nlp-bucket/*"
      ]
    }
  ]
}
```

### Data Protection
- **Encryption at Rest**: S3 server-side encryption
- **Encryption in Transit**: HTTPS for all data transfers
- **Access Control**: IAM policies for resource access
- **Secrets Management**: Environment variables for tokens

## Cost Optimization

### Instance Management
- **Spot Instances**: Up to 70% savings for training workloads
- **Auto-Termination**: Instances terminate after job completion
- **Right-Sizing**: Appropriate instance types for each workload
- **Scheduling**: Training during off-peak hours

### Storage Optimization
- **S3 Intelligent Tiering**: Automatic cost optimization
- **Lifecycle Policies**: Automatic cleanup of old models
- **Compression**: Model files compressed when possible
- **Monitoring**: Cost tracking and alerts

## Monitoring and Logging

### Instance Monitoring
- **CloudWatch Metrics**: CPU, memory, network utilization
- **Custom Metrics**: Training progress, model accuracy
- **Log Aggregation**: Centralized logging via CloudWatch Logs
- **Alerting**: Notifications for failures or completion

### Application Monitoring
- **Health Checks**: Automated service health monitoring
- **Performance Metrics**: Response times, error rates
- **User Analytics**: Usage patterns and feature adoption
- **Model Performance**: Inference latency and accuracy

## Disaster Recovery

### Backup Strategy
- **Model Versioning**: Multiple model versions in S3
- **Code Versioning**: Git-based version control
- **Configuration Backup**: Infrastructure as code
- **Data Backup**: Training data replicated across regions

### Recovery Procedures
- **Instance Replacement**: Automated instance recreation
- **Data Recovery**: Point-in-time recovery from S3
- **Rollback Capability**: Previous model version deployment
- **Service Restoration**: Automated service restoration

## Scaling Considerations

### Horizontal Scaling
- **Multiple Training Instances**: Parallel model training
- **Load Balancing**: Multiple Gradio instances behind ALB
- **Auto Scaling Groups**: Automatic capacity adjustment
- **Regional Deployment**: Multi-region deployment capability

### Vertical Scaling
- **Instance Upgrades**: Larger instance types for heavy workloads
- **Storage Scaling**: Elastic storage capacity
- **Network Optimization**: Enhanced networking for large models
- **GPU Scaling**: Multiple GPU instances for large models

## Performance Optimization

### Training Performance
- **GPU Utilization**: Optimized batch sizes and memory usage
- **Data Loading**: Efficient data pipeline with prefetching
- **Model Checkpointing**: Regular checkpoints for long training
- **Mixed Precision**: FP16 training for speed and memory efficiency

### Inference Performance
- **Model Caching**: Local model caching for faster loading
- **Response Caching**: Cache common responses
- **Connection Pooling**: Efficient network connection management
- **Model Optimization**: Quantization and optimization techniques

## Future Enhancements

### Planned Improvements
- **Container Deployment**: Docker containerization for better portability
- **Kubernetes Integration**: Container orchestration for scalability
- **CI/CD Pipeline**: Automated deployment and testing
- **A/B Testing**: Model comparison and evaluation framework

### Advanced Features
- **Real-time Training**: Continuous model updates
- **Federated Learning**: Distributed training across instances
- **Multi-Model Serving**: Dynamic model loading and switching
- **Advanced Monitoring**: ML-specific monitoring and alerting

This architecture provides a robust, scalable foundation for the Stranger Things AI Analysis Suite with clear separation of concerns and optimized resource utilization.

# EC2/S3 Architecture - Stranger Things NLP Project

## Architecture Overview

The EC2/S3 architecture provides a traditional Infrastructure-as-a-Service (IaaS) approach to deploying the Stranger Things NLP project. This architecture uses manually managed EC2 instances for compute workloads and S3 for data storage, offering maximum control over the infrastructure while requiring more hands-on management.

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          AWS Cloud                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐  │
│  │   VPC Network   │    │   S3 Storage    │    │   IAM Roles  │  │
│  │                 │    │                 │    │              │  │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌──────────┐ │  │
│  │ │ Public      │ │    │ │ Raw Data    │ │    │ │ EC2-S3   │ │  │
│  │ │ Subnet      │ │    │ │ Bucket      │ │    │ │ Access   │ │  │
│  │ │             │ │    │ │             │ │    │ │ Role     │ │  │
│  │ │ ┌─────────┐ │ │    │ ├─────────────┤ │    │ └──────────┘ │  │
│  │ │ │Training │ │ │    │ │ Model       │ │    │ ┌──────────┐ │  │
│  │ │ │Instance │ │ │    │ │ Artifacts   │ │    │ │ Gradio   │ │  │
│  │ │ │         │ │ │    │ │ Bucket      │ │    │ │ Access   │ │  │
│  │ │ │GPU Inst │ │ │    │ │             │ │    │ │ Role     │ │  │
│  │ │ └─────────┘ │ │    │ ├─────────────┤ │    │ └──────────┘ │  │
│  │ │             │ │    │ │ Results     │ │    │              │  │
│  │ │ ┌─────────┐ │ │    │ │ Bucket      │ │    │              │  │
│  │ │ │Gradio   │ │ │    │ │             │ │    │              │  │
│  │ │ │Instance │ │ │◄───┤ └─────────────┘ │    │              │  │
│  │ │ │         │ │ │    │                 │    │              │  │
│  │ │ │Web Host │ │ │    │                 │    │              │  │
│  │ │ └─────────┘ │ │    │                 │    │              │  │
│  │ └─────────────┘ │    │                 │    │              │  │
│  └─────────────────┘    └─────────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         ▲                          ▲                    ▲
         │                          │                    │
    ┌─────────┐              ┌─────────────┐      ┌─────────────┐
    │   SSH   │              │ Data Upload │      │  Web Access │
    │  Access │              │ & Download  │      │  (Port 7860)│
    └─────────┘              └─────────────┘      └─────────────┘
```

## Core Components

### 1. **Compute Layer (EC2 Instances)**

#### Training Instances
- **Purpose**: Execute machine learning training workloads
- **Instance Types**: GPU-enabled (g4dn.xlarge, p3.2xlarge) for model training
- **Configuration**: 
  - Custom AMI with NVIDIA drivers and CUDA toolkit
  - Large EBS volumes (100GB+) for datasets and model storage
  - Auto-scaling disabled (manual scaling)
- **Workloads**:
  - Character chatbot training (Llama, Qwen models)
  - Text classification model fine-tuning
  - Theme classifier training
  - Network analysis processing

#### Gradio Hosting Instances
- **Purpose**: Host web interface for user interactions
- **Instance Types**: CPU-optimized (t3.medium, t3.large) for web serving
- **Configuration**:
  - Standard Ubuntu AMI with web server capabilities
  - Smaller EBS volumes (20-50GB)
  - Public IP with security group allowing HTTP/HTTPS traffic
- **Services**:
  - Gradio web application
  - Model inference endpoints
  - Static file serving

### 2. **Storage Layer (S3 Buckets)**

#### Raw Data Storage
- **Bucket**: `stranger-things-raw-data-{account-id}`
- **Contents**: 
  - Original subtitle files (.srt)
  - Character dialogue datasets
  - Network analysis input data
- **Access**: Read-only for training instances

#### Model Artifacts Storage
- **Bucket**: `stranger-things-models-{account-id}`
- **Contents**:
  - Trained model weights and configurations
  - HuggingFace model artifacts
  - Tokenizers and preprocessors
- **Versioning**: Enabled for model lineage tracking

#### Results and Outputs Storage
- **Bucket**: `stranger-things-results-{account-id}`
- **Contents**:
  - Training logs and metrics
  - Generated network graphs
  - Classification results
  - Inference outputs

### 3. **Network Layer (VPC Configuration)**

#### VPC Setup
- **CIDR**: 10.0.0.0/16
- **Availability Zones**: Multi-AZ for high availability
- **Subnets**:
  - Public subnet (10.0.1.0/24) - Web-facing instances
  - Private subnet (10.0.2.0/24) - Training instances (optional)

#### Security Groups
- **Training Security Group**:
  - Inbound: SSH (22) from admin IPs
  - Outbound: HTTPS (443) for model downloads, S3 access
- **Web Security Group**:
  - Inbound: SSH (22), HTTP (80), HTTPS (443), Gradio (7860)
  - Outbound: HTTPS (443) for API calls, S3 access

### 4. **Identity and Access Management (IAM)**

#### EC2-S3 Access Role
- **Policies Attached**:
  - `AmazonS3FullAccess` (scoped to project buckets)
  - `AmazonEC2ReadOnlyAccess` (for instance self-discovery)
  - Custom policy for HuggingFace Hub access

#### Service-Specific Policies
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::stranger-things-*",
        "arn:aws:s3:::stranger-things-*/*"
      ]
    }
  ]
}
```

## Data Flow Architecture

### 1. **Training Pipeline Data Flow**

```
Local Development → S3 Raw Data → EC2 Training Instance → S3 Model Artifacts
     │                 ▲               │                    │
     │                 │               ▼                    ▼
     └─── Code Sync ────┘         Training Logs      Model Registry
                                       │                    │
                                       ▼                    │
                                CloudWatch Logs             │
                                                            │
                             ┌──────────────────────────────┘
                             │
                             ▼
                    Gradio Instance ← S3 Model Download
```

### 2. **Inference Pipeline Data Flow**

```
User Request → Gradio Web Interface → Model Inference → Response
     │              │                      │              │
     │              ▼                      ▼              │
     │        Load Balancer           Model Cache          │
     │              │                      │              │
     │              ▼                      ▼              │
     └─────── Security Layer ───── Resource Monitor ──────┘
```

### 3. **Development Workflow**

```
Local Development
     │
     ├── Code Changes → Git Repository
     │                      │
     ├── Data Preparation   ▼
     │       │         EC2 Deployment Script
     │       ▼              │
     ├── S3 Upload          ▼
     │       │         Instance Launch
     │       │              │
     │       └──────────────▼
     └── Training Job → Model Training → Model Artifacts → Deployment
```

## Component Responsibilities

### AWS Infrastructure (`aws/` directory)
- **config.py**: Centralized configuration management for AWS resources
- **ec2_manager.py**: EC2 instance lifecycle management, deployment automation
- **storage.py**: S3 operations, data synchronization, model artifact management
- **ec2_orchestrator.py**: High-level orchestration of training and deployment workflows

### Core NLP Components
- **character_chatbot/**: Chatbot training and inference logic
- **text_classification/**: Fine-tuning pipelines for classification models
- **theme_classifier/**: Zero-shot theme classification using pre-trained models
- **character_network/**: Named entity recognition and network graph generation
- **utils/**: Shared utilities for data loading, preprocessing

### Application Layer
- **gradio_app.py**: Web interface combining all NLP components
- **deploy_aws.py**: Deployment orchestration script

## Deployment Strategies

### 1. **On-Demand Deployment**
- Launch instances when needed
- Terminate after completion
- Cost-effective for irregular workloads
- Higher latency for cold starts

### 2. **Persistent Deployment**
- Keep instances running
- Faster response times
- Higher costs for continuous operation
- Suitable for production workloads

### 3. **Spot Instance Strategy**
- Use spot instances for training (up to 70% cost savings)
- Implement checkpoint-based training for interruption handling
- Reserved instances for critical production services

## Scalability Considerations

### Horizontal Scaling
- **Auto Scaling Groups**: Not implemented in basic version
- **Load Balancers**: Application Load Balancer for Gradio instances
- **Multi-AZ Deployment**: Distribute instances across availability zones

### Vertical Scaling
- **Instance Resizing**: Upgrade instance types as needed
- **Storage Scaling**: Expand EBS volumes during operation
- **Memory Optimization**: Choose memory-optimized instances for large models

## Cost Optimization Strategies

### 1. **Compute Optimization**
- Use spot instances for training workloads
- Implement automatic instance termination after job completion
- Schedule training jobs during off-peak hours
- Use smaller instances for development/testing

### 2. **Storage Optimization**
- Implement S3 lifecycle policies
- Use intelligent tiering for infrequently accessed data
- Compress model artifacts before storage
- Delete temporary files and logs regularly

### 3. **Network Optimization**
- Place compute and storage in same region
- Use VPC endpoints for S3 access to avoid NAT gateway costs
- Implement CloudFront for static web content

## Monitoring and Logging

### CloudWatch Integration
- **Custom Metrics**: Training progress, model accuracy, inference latency
- **Log Groups**: Separate log groups for each component
- **Alarms**: Cost alerts, instance health checks, disk usage warnings
- **Dashboards**: Real-time monitoring of system performance

### Application-Level Monitoring
- Training job progress tracking
- Model performance metrics
- User interaction analytics
- Resource utilization monitoring

## Security Considerations

### 1. **Access Control**
- SSH key-based authentication
- Security groups with minimal required ports
- IAM roles with least-privilege access
- Regular security group audits

### 2. **Data Protection**
- S3 bucket encryption at rest
- EBS volume encryption
- SSL/TLS for data in transit
- Regular backup and disaster recovery testing

### 3. **Network Security**
- VPC with private subnets for sensitive workloads
- NACLs for additional network layer security
- VPN or Direct Connect for secure on-premises connectivity
- Regular security patches and updates

## Disaster Recovery

### Backup Strategy
- **Daily S3 snapshots**: Automated backup of model artifacts
- **EBS snapshots**: Regular snapshots of instance root volumes
- **Cross-region replication**: Critical data replicated to secondary region
- **Code repository**: All code version controlled in Git

### Recovery Procedures
- **Instance Recovery**: Launch new instances from AMI snapshots
- **Data Recovery**: Restore from S3 backups and EBS snapshots
- **Application Recovery**: Redeploy from version-controlled code
- **Testing**: Regular disaster recovery drills

## Migration Path to SageMaker

The EC2/S3 architecture serves as a foundation that can be migrated to SageMaker for enhanced scalability and reduced operational overhead:

1. **Data Migration**: S3 buckets can be directly used by SageMaker
2. **Model Migration**: Existing model artifacts compatible with SageMaker
3. **Code Migration**: Training scripts can be containerized for SageMaker
4. **Gradual Transition**: Run both architectures in parallel during migration
5. **Feature Parity**: Ensure all EC2-based features are available in SageMaker version

This architecture provides a solid foundation for ML workloads while maintaining full control over the infrastructure and enabling future migration to more managed services like SageMaker.