# Custom SageMaker Training Container for Stranger Things NLP
# This container includes all dependencies for training character chatbots and text classifiers

FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install core ML dependencies
RUN pip3 install \
    transformers==4.30.0 \
    datasets==2.12.0 \
    tokenizers==0.13.3 \
    accelerate==0.20.3 \
    peft==0.3.0 \
    trl==0.4.7 \
    bitsandbytes==0.39.1 \
    scipy==1.10.1 \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    huggingface-hub==0.15.1

# Install additional NLP dependencies
RUN pip3 install \
    spacy==3.6.0 \
    nltk==3.8.1 \
    evaluate==0.4.0 \
    rouge-score==0.1.2

# Install AWS and SageMaker dependencies
RUN pip3 install \
    boto3==1.28.0 \
    sagemaker==2.165.0 \
    sagemaker-training==4.6.0

# Install other utilities
RUN pip3 install \
    tqdm==4.65.0 \
    wandb==0.15.4 \
    tensorboard==2.13.0 \
    jupyter==1.0.0 \
    ipywidgets==8.0.6

# Download spacy model
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en_core_web_trf

# Set working directory
WORKDIR /opt/ml

# Create necessary directories for SageMaker
RUN mkdir -p /opt/ml/input/data \
    && mkdir -p /opt/ml/input/config \
    && mkdir -p /opt/ml/model \
    && mkdir -p /opt/ml/output \
    && mkdir -p /opt/ml/code \
    && mkdir -p /opt/ml/checkpoints

# Copy training scripts
COPY train.py /opt/ml/code/train.py
COPY inference.py /opt/ml/code/inference.py
COPY requirements.txt /opt/ml/code/requirements.txt

# Make training script executable
RUN chmod +x /opt/ml/code/train.py

# Set environment variables for SageMaker
ENV SAGEMAKER_PROGRAM=train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_REGION=us-east-1

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default entry point for training
ENTRYPOINT ["python3", "/opt/ml/code/train.py"]