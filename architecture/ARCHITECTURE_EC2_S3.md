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