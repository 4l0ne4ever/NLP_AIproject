# Stranger Things NLP - Complete Demo Flow

This guide walks through the complete training and deployment process for the Stranger Things AI character chatbot system.

## Prerequisites

- AWS CLI configured with appropriate permissions
- EC2 key pair created (`stranger-things-key-dct`)
- S3 bucket accessible (`stranger-things-nlp-duongcongthuyet`)
- Python 3.8+ with virtual environment

---

## Pre-flight Checks

Verify your AWS setup and data availability:

```bash
# Verify AWS identity and region
aws sts get-caller-identity
aws configure get region

# Check training data is available in S3
aws s3 ls s3://stranger-things-nlp-duongcongthuyet/data/training/transcripts/ --recursive | head -10

# Ensure SSH key has correct permissions
chmod 400 ~/.ssh/stranger-things-key-dct.pem

# Navigate to project directory
cd "/Users/duongcongthuyet/Downloads/workspace/AI /project"
source venv/bin/activate
```

**Expected outputs:**
- AWS account ID and region (should be `us-east-1`)
- ~34 transcript CSV files in S3
- Key permissions set to 400

---

## Step 1: Start LLaMA Training

Launch EC2 GPU training for the LLaMA model:

```bash
# Start LLaMA training (optimized config: 800 steps, ~45-60 minutes)
python3 deploy_aws.py train llama
```

**Expected output:**
```
AWS managers initialized successfully
Starting LLAMA training on EC2...
Training instance launched: 44.200.84.242
Job name: llama-training-1758887197
Deploying code to instance...
Code deployed successfully
Starting training process...
Training started successfully!
SSH command: ssh -i ~/.ssh/stranger-things-key-dct.pem ubuntu@44.200.84.242
Monitor training: tail -f training.log
```

**Save these details:**
- **Instance IP**: `44.200.84.242` (example)
- **Job name**: `llama-training-1758887197` (example)
- **SSH command** for monitoring

---

## Step 2: Monitor LLaMA Training Progress

Track training progress in real-time:

```bash
# Replace <IP> with your actual instance IP from Step 1
export LLAMA_IP="44.200.84.242"

# Monitor training logs (real-time)
ssh -o StrictHostKeyChecking=no -i ~/.ssh/stranger-things-key-dct.pem ubuntu@$LLAMA_IP "tail -f /home/ubuntu/training.log"

# Alternative: Check last 50 lines
ssh -o StrictHostKeyChecking=no -i ~/.ssh/stranger-things-key-dct.pem ubuntu@$LLAMA_IP "tail -n 50 /home/ubuntu/training.log || echo 'Log not ready yet'"
```

**What to look for:**
```
✓ Authenticating with HuggingFace...
✓ CUDA Available: True
✓ GPU Name: Tesla T4
✓ Downloaded 34 files to /tmp/stranger_things_models/transcripts
✓ Processed 10924 dialogue samples  
✓ 📡 S3 live checkpoint saving enabled
✓ Training progress: 150/800 [18%] 🕒 ETA: 35min
```

**Training will complete when you see:**
```
✓ Training pipeline completed!
✓ Model uploaded to S3: models/trained/llama/20250926-091132/
✓ Run ID: 20250926-091132
```

---

## Step 3: Start Qwen Training (After LLaMA Completes)

Launch the second model training:

```bash
# This will fail if LLaMA is still running (by design)
python3 deploy_aws.py train qwen

# If LLaMA is still running, you'll see:
# "Error: An active training instance already exists (i-xxxxx). 
#  Only one training job is allowed at a time."

# Wait for LLaMA to complete, then retry
```

**Monitor Qwen training similarly:**
```bash
export QWEN_IP="<new_instance_ip>"
ssh -o StrictHostKeyChecking=no -i ~/.ssh/stranger-things-key-dct.pem ubuntu@$QWEN_IP "tail -f /home/ubuntu/training.log"
```

---

## Step 4: Check Instance Status

Monitor all running instances:

```bash
python3 deploy_aws.py list-instances
```

**Expected output during training:**
```
Training Instances:
  [ACTIVE] llama-training-1758887197: 44.200.84.242 (running)

Gradio Instances:
  [IDLE] stranger-things-gradio: (stopped)
```

**After training completes:**
```
Training Instances:
  [IDLE] llama-training-1758887197: (stopped)
  [IDLE] qwen-training-1758889543: (stopped)
```

---

## Step 5: Verify Training Results in S3

Check that models were successfully uploaded to S3:

### LLaMA Model Results:
```bash
# List all LLaMA model files
aws s3 ls s3://stranger-things-nlp-duongcongthuyet/models/trained/llama/ --recursive | sort | tail -10

# Get latest model metadata
aws s3 cp s3://stranger-things-nlp-duongcongthuyet/models/trained/llama/latest.json ./llama_latest.json
cat ./llama_latest.json | jq '.'
```

### Qwen Model Results:
```bash
# List all Qwen model files  
aws s3 ls s3://stranger-things-nlp-duongcongthuyet/models/trained/qwen/ --recursive | sort | tail -10

# Get latest model metadata
aws s3 cp s3://stranger-things-nlp-duongcongthuyet/models/trained/qwen/latest.json ./qwen_latest.json
cat ./qwen_latest.json | jq '.'
```

### Training Logs and Checkpoints:
```bash
# Check training logs
aws s3 ls s3://stranger-things-nlp-duongcongthuyet/logs/llama/ --recursive | head -5
aws s3 ls s3://stranger-things-nlp-duongcongthuyet/logs/qwen/ --recursive | head -5

# Check saved checkpoints
aws s3 ls s3://stranger-things-nlp-duongcongthuyet/checkpoints/llama/ --recursive | head -5
aws s3 ls s3://stranger-things-nlp-duongcongthuyet/live_checkpoints/ --recursive | head -5
```

**Expected model metadata structure:**
```json
{
  "model_path": "models/trained/llama/20250926-091132/",
  "timestamp": "20250926-091132",
  "accuracy": 0.85,
  "loss": 0.32,
  "training_data": "Stranger Things transcripts",
  "model_type": "llama",
  "base_model": "meta-llama/Llama-3.2-3B-Instruct",
  "training_config": {
    "batch_size": 8,
    "max_steps": 800,
    "learning_rate": 0.0002,
    "max_length": 256
  }
}
```

---

## Step 6: Local Demo Application

Run the Gradio app locally to test the trained models:

### Setup:
```bash
# Ensure dependencies are installed
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Optional: Pre-cache the latest LLaMA model for faster loading
RUN_ID=$(python3 -c "import json; print(json.load(open('llama_latest.json'))['timestamp'])" 2>/dev/null || echo "fallback")
if [ "$RUN_ID" != "fallback" ]; then
    echo "Pre-caching LLaMA model: $RUN_ID"
    aws s3 sync s3://stranger-things-nlp-duongcongthuyet/models/trained/llama/$RUN_ID/ /tmp/stranger_things_models/llama/$RUN_ID/
else
    echo "No trained model found, will use HuggingFace fallback"
fi
```

### Launch Application:
```bash
# Start the Gradio app with fallback announcement system
python3 gradio_app_v2.py
```

**Expected output:**
```
Loading Stranger Things AI Analysis Suite...
✓ Theme classifier loaded
✓ Character network analyzer loaded  
✓ Location classifier loaded
✓ Checking for trained models in S3...
✓ Found trained LLaMA model: 20250926-091132
✓ Found trained Qwen model: 20250926-112045
Running on local URL:  http://127.0.0.1:7860
```

### Open the application:
```bash
# Open in your default browser
open http://localhost:7860
```

---

## Step 7: Demo Talking Points

When demonstrating the application, highlight these features:

### Smart Model Management:
- **"The app automatically detects trained models in S3"**
- **"If no trained model is available, it falls back to HuggingFace pre-trained models"**
- **"Notice messages inform users when fallback models are being used"**

### Training Optimizations:
- **"We trained on AWS EC2 with GPU optimization (Tesla T4)"**
- **"Training time reduced from 3+ hours to ~45-60 minutes per model"**
- **"Used 800 training steps with batch size 8 and FP16 mixed precision"**
- **"Real-time progress monitoring every 10 steps vs. previous 100+ step intervals"**

### Cloud Architecture:
- **"Complete AWS deployment: EC2 for compute, S3 for storage"**
- **"Live checkpoint saving during training - no progress lost"**
- **"All artifacts saved: datasets, checkpoints, logs, final models"**
- **"EC2 instances auto-terminate after training to control costs"**

### Application Features:
1. **Character Chatbots** - Chat with Eleven, Mike, Dustin using fine-tuned models
2. **Theme Classification** - Analyze text for themes like friendship, mystery, supernatural
3. **Character Network** - Interactive relationship visualizations
4. **Location Classification** - Classify text locations using custom models

---

## Step 8: Cleanup

Clean up resources to avoid unnecessary costs:

### Check what's running:
```bash
python3 deploy_aws.py list-instances
python3 deploy_aws.py status
```

### Terminate instances:
```bash
# Get instance IDs from list-instances output
python3 deploy_aws.py terminate i-0123456789abcdef0
python3 deploy_aws.py terminate i-0987654321fedcba0

# Verify termination
python3 deploy_aws.py list-instances
```

### Alternative - terminate all running instances:
```bash
# Emergency cleanup (use with caution)
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" --query "Reservations[*].Instances[*].InstanceId" --output text | xargs -r aws ec2 terminate-instances --instance-ids
```

---

## Troubleshooting

### SSH Connection Issues:
```bash
# Verify key permissions
ls -la ~/.ssh/stranger-things-key-dct.pem  # Should show -r--------

# Test connectivity 
nc -vz <PUBLIC_IP> 22

# Debug SSH connection
ssh -vvv -i ~/.ssh/stranger-things-key-dct.pem ubuntu@<PUBLIC_IP>
```

### Training Not Starting:
```bash
# Check instance status
aws ec2 describe-instances --instance-ids i-<your-instance-id>

# Check if training process is running
ssh -i ~/.ssh/stranger-things-key-dct.pem ubuntu@<IP> "ps aux | grep python"

# Check system logs
ssh -i ~/.ssh/stranger-things-key-dct.pem ubuntu@<IP> "sudo tail -50 /var/log/cloud-init-output.log"
```

### Model Not Loading in Gradio:
```bash
# Check S3 model availability
aws s3 ls s3://stranger-things-nlp-duongcongthuyet/models/trained/ --recursive

# Check local cache
ls -la /tmp/stranger_things_models/

# Test HuggingFace fallback
python3 -c "from transformers import pipeline; print('HF access working')"
```

---

## Performance Benchmarks

### Training Time Comparison:
- **Before optimization**: 2-3+ hours, often failed
- **After optimization**: 45-60 minutes, reliable completion
- **Key improvements**: GPU usage, batch optimization, sequence length reduction

### Cost Estimates:
- **Training**: ~$0.50/hour × 1 hour = $0.50 per model
- **S3 storage**: ~$0.023/GB/month for models and logs
- **Total for both models**: ~$1.00 + storage costs

### Model Quality:
- **LLaMA-3.2-3B**: Better character personality capture
- **Qwen-3B**: Higher general accuracy, faster inference
- **Training data**: 10,924 dialogue samples from Stranger Things seasons 1-4

---

## Next Steps

After completing this demo:

1. **Deploy to production**: Use `python3 deploy_aws.py deploy-gradio` for public hosting
2. **Scale training**: Experiment with larger models or more training steps  
3. **Add new characters**: Extend training data for additional character personalities
4. **Integrate APIs**: Connect to external applications or chatbot platforms
5. **Monitor usage**: Set up CloudWatch monitoring for production deployments

---

*This demo showcases a complete MLOps pipeline: data processing, distributed training, model management, and deployment - all orchestrated through AWS infrastructure with intelligent fallback capabilities.*