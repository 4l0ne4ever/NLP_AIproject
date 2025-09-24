#!/bin/bash
# Build and Push Custom SageMaker Training Container
# This script builds the Docker container and pushes it to Amazon ECR

set -e

# Configuration
CONTAINER_NAME="stranger-things-training"
TAG="latest"
REGION=${AWS_DEFAULT_REGION:-us-east-1}
ACCOUNT_ID=${AWS_ACCOUNT_ID}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed or not in PATH"
        exit 1
    fi
    
    # Check if AWS credentials are configured
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Run 'aws configure'"
        exit 1
    fi
    
    # Check if ACCOUNT_ID is set
    if [[ -z "$ACCOUNT_ID" ]]; then
        log_info "AWS_ACCOUNT_ID not set, attempting to retrieve..."
        ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
        if [[ -z "$ACCOUNT_ID" ]]; then
            log_error "Could not retrieve AWS Account ID"
            exit 1
        fi
        log_success "Retrieved AWS Account ID: $ACCOUNT_ID"
    fi
    
    log_success "Prerequisites check completed"
}

# Create ECR repository if it doesn't exist
create_ecr_repository() {
    log_info "Creating ECR repository if it doesn't exist..."
    
    # Check if repository exists
    if aws ecr describe-repositories --repository-names $CONTAINER_NAME --region $REGION &> /dev/null; then
        log_info "ECR repository '$CONTAINER_NAME' already exists"
    else
        log_info "Creating ECR repository '$CONTAINER_NAME'..."
        aws ecr create-repository --repository-name $CONTAINER_NAME --region $REGION
        log_success "ECR repository created successfully"
    fi
}

# Get ECR login token
ecr_login() {
    log_info "Logging in to Amazon ECR..."
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
    log_success "Successfully logged in to ECR"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    # Full image name
    FULL_IMAGE_NAME="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$CONTAINER_NAME:$TAG"
    
    # Build the image
    docker build \
        --platform linux/amd64 \
        -t $CONTAINER_NAME:$TAG \
        -t $FULL_IMAGE_NAME \
        .
    
    log_success "Docker image built successfully: $FULL_IMAGE_NAME"
}

# Push image to ECR
push_image() {
    log_info "Pushing image to Amazon ECR..."
    
    FULL_IMAGE_NAME="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$CONTAINER_NAME:$TAG"
    
    docker push $FULL_IMAGE_NAME
    
    log_success "Image pushed successfully to ECR"
    log_success "Image URI: $FULL_IMAGE_NAME"
}

# Test the container locally (optional)
test_container() {
    log_info "Testing container locally..."
    
    # Create test directories
    mkdir -p test_data/input/data/train
    mkdir -p test_data/model
    mkdir -p test_data/output
    
    # Create a simple test file
    echo '{"prompt": "Hello, this is a test"}' > test_data/input/data/train/test.json
    
    # Run container for a quick test
    docker run --rm \
        -v $(pwd)/test_data:/opt/ml \
        -e SAGEMAKER_PROGRAM=train.py \
        $CONTAINER_NAME:$TAG \
        --test-mode || log_warning "Container test failed (this may be expected if no test data is provided)"
    
    # Clean up test data
    rm -rf test_data
    
    log_info "Container test completed"
}

# Update SageMaker configuration with new image
update_sagemaker_config() {
    log_info "Updating SageMaker configuration..."
    
    FULL_IMAGE_NAME="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$CONTAINER_NAME:$TAG"
    
    # Create a config update script
    cat > update_config.py << EOF
import sys
sys.path.append('..')
from config import SageMakerConfigManager

# Update training image in config
config = SageMakerConfigManager()
config.training_config.training_image = '$FULL_IMAGE_NAME'
config.save_config()

print(f"✅ Updated SageMaker config with image: $FULL_IMAGE_NAME")
EOF
    
    python3 update_config.py
    rm update_config.py
    
    log_success "SageMaker configuration updated"
}

# Main execution
main() {
    echo "🐋 Building and Pushing SageMaker Training Container"
    echo "=================================================="
    echo "Container: $CONTAINER_NAME"
    echo "Tag: $TAG"
    echo "Region: $REGION"
    echo "Account: $ACCOUNT_ID"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Create ECR repository
    create_ecr_repository
    
    # Login to ECR
    ecr_login
    
    # Build Docker image
    build_image
    
    # Test container (optional, comment out if not needed)
    if [[ "$1" == "--test" ]]; then
        test_container
    fi
    
    # Push image to ECR
    push_image
    
    # Update SageMaker configuration
    update_sagemaker_config
    
    echo ""
    log_success "🎉 Container build and push completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Your custom training container is now available in ECR"
    echo "2. The SageMaker configuration has been updated to use this image"
    echo "3. You can now launch training jobs using: python ../deploy.py train"
    echo ""
    echo "🔗 ECR Repository: https://$REGION.console.aws.amazon.com/ecr/repositories/private/$ACCOUNT_ID/$CONTAINER_NAME"
    echo "📊 Image URI: $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$CONTAINER_NAME:$TAG"
}

# Handle script arguments
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 [--test] [--help]"
    echo ""
    echo "Options:"
    echo "  --test    Run container tests after building"
    echo "  --help    Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  AWS_DEFAULT_REGION    AWS region (default: us-east-1)"
    echo "  AWS_ACCOUNT_ID        AWS account ID (auto-detected if not set)"
    exit 0
fi

# Change to script directory
cd "$(dirname "$0")"

# Run main function
main "$@"