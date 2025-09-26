import boto3
import time
import json
import logging
import paramiko
import socket
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from botocore.exceptions import ClientError
from .config import AWSConfigManager
from .storage import StrangerThingsS3Manager

class EC2TrainingOrchestrator:
    """Orchestrate EC2 instances for training and Gradio app deployment"""
    
    def __init__(self, config_manager: AWSConfigManager):
        self.config = config_manager
        self.ec2_client = boto3.client('ec2', region_name=self.config.ec2_config.region)
        self.ec2_resource = boto3.resource('ec2', region_name=self.config.ec2_config.region)
        self.s3_manager = StrangerThingsS3Manager(
            bucket_name=self.config.s3_config.bucket_name
        )
        
        # Track running instances
        self.training_instances = {}
        self.gradio_instances = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_security_group(self, name: str = "stranger-things-sg") -> str:
        """Create security group for EC2 instances"""
        try:
            # Check if security group already exists
            existing_sgs = self.ec2_client.describe_security_groups(
                Filters=[{'Name': 'group-name', 'Values': [name]}]
            )
            
            if existing_sgs['SecurityGroups']:
                sg_id = existing_sgs['SecurityGroups'][0]['GroupId']
                self.logger.info(f"Using existing security group: {sg_id}")
                return sg_id
            
            # Create new security group
            response = self.ec2_client.create_security_group(
                GroupName=name,
                Description="Security group for Stranger Things NLP project"
            )
            
            sg_id = response['GroupId']
            
            # Add rules for SSH and Gradio
            self.ec2_client.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 22,
                        'ToPort': 22,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': self.config.gradio_config.port,
                        'ToPort': self.config.gradio_config.port,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 80,
                        'ToPort': 80,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 443,
                        'ToPort': 443,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )
            
            self.logger.info(f"Created security group: {sg_id}")
            return sg_id
            
        except ClientError as e:
            self.logger.error(f"Error creating security group: {e}")
            raise
    
    def launch_training_instance(self, 
                                model_type: str = "llama",
                                training_job_name: str = None) -> Dict:
        """Launch EC2 instance for model training"""
        
        if training_job_name is None:
            training_job_name = f"training-{model_type}-{int(time.time())}"
        
        # Enforce single active training job at a time
        active_training = self._find_active_training_instances()
        if active_training:
            active_ids = ", ".join([i['instance_id'] for i in active_training])
            raise Exception(
                f"An active training instance already exists ({active_ids}). "
                f"Only one training job is allowed at a time. Terminate the existing job to proceed."
            )
        
        # Ensure security group exists
        sg_id = self.create_security_group()
        
        # Prepare user data script
        user_data = self.config.get_ec2_user_data_script()
        
        try:
            # Launch instance
            if self.config.ec2_config.use_spot_instances:
                response = self.ec2_client.request_spot_instances(
                    SpotPrice=str(self.config.ec2_config.max_spot_price),
                    InstanceCount=1,
                    Type='one-time',
                    LaunchSpecification={
                        'ImageId': self.config.ec2_config.ami_id,
                        'InstanceType': self.config.ec2_config.instance_type,
                        'KeyName': self.config.ec2_config.key_pair_name,
                        'SecurityGroupIds': [sg_id],
                        'UserData': user_data,
                        'BlockDeviceMappings': [
                            {
                                'DeviceName': '/dev/sda1',
                                'Ebs': {
                                    'VolumeSize': self.config.ec2_config.volume_size,
                                    'VolumeType': self.config.ec2_config.volume_type,
                                    'DeleteOnTermination': True
                                }
                            },
                            # Additional volume for model storage and training data
                            {
                                'DeviceName': '/dev/sdf',
                                'Ebs': {
                                    'VolumeSize': 200,  # 200GB for models and training data
                                    'VolumeType': 'gp3',
                                    'DeleteOnTermination': True,
                                    'Encrypted': True
                                }
                            },
                            # Additional volume for cache and temporary files
                            {
                                'DeviceName': '/dev/sdg',
                                'Ebs': {
                                    'VolumeSize': 100,  # 100GB for cache
                                    'VolumeType': 'gp3',
                                    'DeleteOnTermination': True,
                                    'Encrypted': True
                                }
                            }
                        ],
                        'IamInstanceProfile': {
                            'Name': 'EC2-S3-Access'  # Assume this role exists
                        }
                    }
                )
                
                spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
                self.logger.info(f"Launched spot instance request: {spot_request_id}")
                
                # Wait for spot instance to be fulfilled
                instance_id = self._wait_for_spot_instance(spot_request_id)
                
                # Tag the instance after fulfillment (spot tagging)
                try:
                    self.ec2_client.create_tags(
                        Resources=[instance_id],
                        Tags=[
                            {'Key': 'Name', 'Value': f'stranger-things-training-{model_type}'},
                            {'Key': 'Project', 'Value': 'stranger-things-nlp'},
                            {'Key': 'Purpose', 'Value': 'training'},
                            {'Key': 'ModelType', 'Value': model_type},
                            {'Key': 'JobName', 'Value': training_job_name}
                        ]
                    )
                except ClientError as e:
                    self.logger.warning(f"Failed to tag spot instance {instance_id}: {e}")
                
            else:
                response = self.ec2_client.run_instances(
                    ImageId=self.config.ec2_config.ami_id,
                    MinCount=1,
                    MaxCount=1,
                    InstanceType=self.config.ec2_config.instance_type,
                    KeyName=self.config.ec2_config.key_pair_name,
                    SecurityGroupIds=[sg_id],
                    UserData=user_data,
                    BlockDeviceMappings=[
                        {
                            'DeviceName': '/dev/sda1',
                            'Ebs': {
                                'VolumeSize': self.config.ec2_config.volume_size,
                                'VolumeType': self.config.ec2_config.volume_type,
                                'DeleteOnTermination': True
                            }
                        },
                        # Additional volume for model storage and training data
                        {
                            'DeviceName': '/dev/sdf',
                            'Ebs': {
                                'VolumeSize': 200,  # 200GB for models and training data
                                'VolumeType': 'gp3',
                                'DeleteOnTermination': True,
                                'Encrypted': True
                            }
                        },
                        # Additional volume for cache and temporary files
                        {
                            'DeviceName': '/dev/sdg',
                            'Ebs': {
                                'VolumeSize': 100,  # 100GB for cache
                                'VolumeType': 'gp3',
                                'DeleteOnTermination': True,
                                'Encrypted': True
                            }
                        }
                    ],
                    IamInstanceProfile={
                        'Name': 'EC2-S3-Access'  # Assume this role exists
                    },
                    TagSpecifications=[
                        {
                            'ResourceType': 'instance',
                            'Tags': [
                                {'Key': 'Name', 'Value': f'stranger-things-training-{model_type}'},
                                {'Key': 'Project', 'Value': 'stranger-things-nlp'},
                                {'Key': 'Purpose', 'Value': 'training'},
                                {'Key': 'ModelType', 'Value': model_type},
                                {'Key': 'JobName', 'Value': training_job_name}
                            ]
                        }
                    ]
                )
                
                instance_id = response['Instances'][0]['InstanceId']
                self.logger.info(f"Launched on-demand instance: {instance_id}")
            
            # Wait for instance to be running
            self.logger.info("Waiting for instance to be running...")
            waiter = self.ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])

            # Wait for system and instance status checks to pass
            self._wait_for_instance_status_ok(instance_id)
            
            # Get instance details (ensure we have a public IP)
            instance_info = self._wait_for_public_ip(instance_id)
            
            # Store training job info
            self.training_instances[training_job_name] = {
                'instance_id': instance_id,
                'model_type': model_type,
                'public_ip': instance_info['public_ip'],
                'private_ip': instance_info['private_ip'],
                'launch_time': time.time(),
                'status': 'initializing'
            }
            
            # Also make sure SSH is reachable before returning
            try:
                self._wait_for_ssh(instance_info['public_ip'])
            except Exception as e:
                self.logger.warning(f"SSH not yet reachable but continuing: {e}")
            
            self.logger.info(f"Training instance ready: {instance_info['public_ip']}")
            return self.training_instances[training_job_name]
            
        except ClientError as e:
            self.logger.error(f"Error launching training instance: {e}")
            raise
    
    def launch_gradio_instance(self) -> Dict:
        """Launch EC2 instance for Gradio app hosting"""
        
        sg_id = self.create_security_group()
        
        # Modified user data for Gradio hosting
        user_data = self._get_gradio_user_data_script()
        
        try:
            response = self.ec2_client.run_instances(
                ImageId=self.config.ec2_config.ami_id,
                MinCount=1,
                MaxCount=1,
                InstanceType="t3.medium",  # Smaller instance for hosting
                KeyName=self.config.ec2_config.key_pair_name,
                SecurityGroupIds=[sg_id],
                UserData=user_data,
                IamInstanceProfile={
                    'Name': 'EC2-S3-Access'
                },
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': 'stranger-things-gradio'},
                            {'Key': 'Project', 'Value': 'stranger-things-nlp'},
                            {'Key': 'Purpose', 'Value': 'gradio-hosting'}
                        ]
                    }
                ]
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            
            # Wait for instance to be running
            waiter = self.ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])

            # Wait for system and instance status checks to pass
            self._wait_for_instance_status_ok(instance_id)
            
            # Ensure public IP is available
            instance_info = self._wait_for_public_ip(instance_id)
            
            self.gradio_instances['main'] = {
                'instance_id': instance_id,
                'public_ip': instance_info['public_ip'],
                'private_ip': instance_info['private_ip'],
                'launch_time': time.time(),
                'status': 'initializing',
                'gradio_url': f"http://{instance_info['public_ip']}:{self.config.gradio_config.port}"
            }
            
            # Warm up SSH reachability
            try:
                self._wait_for_ssh(instance_info['public_ip'])
            except Exception as e:
                self.logger.warning(f"SSH not yet reachable but continuing: {e}")
            
            self.logger.info(f"Gradio instance ready: {instance_info['public_ip']}")
            return self.gradio_instances['main']
            
        except ClientError as e:
            self.logger.error(f"Error launching Gradio instance: {e}")
            raise
    
    def deploy_code_to_instance(self, instance_info: Dict, 
                               local_project_path: str = ".") -> bool:
        """Deploy code to EC2 instance via SCP"""
        try:
            # Wait for SSH to be reachable before attempting connection
            self._wait_for_ssh(instance_info['public_ip'])

            # Create SSH client
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect to instance
            key_path = Path(f"~/.ssh/{self.config.ec2_config.key_pair_name}.pem").expanduser()
            
            # Load private key
            try:
                from paramiko import RSAKey, Ed25519Key, ECDSAKey
                # Try different key types
                private_key = None
                for key_class in [RSAKey, Ed25519Key, ECDSAKey]:
                    try:
                        private_key = key_class.from_private_key_file(str(key_path))
                        break
                    except Exception:
                        continue
                
                if private_key is None:
                    raise Exception(f"Could not load private key from {key_path}")
                
                ssh_client.connect(
                    hostname=instance_info['public_ip'],
                    username='ubuntu',  # Ubuntu AMI uses ubuntu
                    pkey=private_key
                )
            except Exception as e:
                # Fallback to key_filename method
                self.logger.warning(f"Failed to load key manually: {e}, trying key_filename method")
                ssh_client.connect(
                    hostname=instance_info['public_ip'],
                    username='ubuntu',  # Ubuntu AMI uses ubuntu
                    key_filename=str(key_path)
                )
            
            # Create base directory first
            ssh_client.exec_command(f"mkdir -p /home/ubuntu/stranger-things-nlp")
            
            # Upload project files
            sftp = ssh_client.open_sftp()
            
            local_path = Path(local_project_path)
            remote_base = "/home/ubuntu/stranger-things-nlp"
            
            # Ensure remote base directory exists
            try:
                sftp.mkdir(remote_base)
            except Exception:
                pass  # Directory might already exist
            
            # Upload essential files
            essential_files = [
                'requirements.txt',
                '.env',                  # Add environment variables
                'gradio_app.py',
                'training_pipeline.py',  # Add training pipeline
                'config.py',             # Add config file
                'aws/',
                'character_chatbot/',
                'text_classification/',
                'theme_classifier/',
                'character_network/',
                'utils/'
            ]
            
            for item in essential_files:
                local_item = local_path / item
                if local_item.exists():
                    if local_item.is_file():
                        remote_path = f"{remote_base}/{item}"
                        sftp.put(str(local_item), remote_path)
                        self.logger.info(f"Uploaded {item}")
                    else:
                        # Upload directory recursively
                        self._upload_directory_recursive(sftp, str(local_item), f"{remote_base}/{item}")
            
            sftp.close()
            
            # Install requirements will be handled in training script
            commands = [
                "cd /home/ubuntu/stranger-things-nlp",
                "echo 'Code deployed, venv will be created during training'"
            ]
            
            for cmd in commands:
                stdin, stdout, stderr = ssh_client.exec_command(cmd)
                stdout.read()  # Wait for completion
            
            ssh_client.close()
            self.logger.info("Code deployment completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying code: {e}")
            return False
    
    def start_training(self, training_job_name: str, model_type: str = "llama", stream_logs: bool = False, log_lines: int = 200, s3_transcripts_prefix: Optional[str] = None) -> bool:
        """Start training on EC2 instance. If stream_logs is True, tail the training log live."""
        if training_job_name not in self.training_instances:
            self.logger.error(f"Training job not found: {training_job_name}")
            return False
        
        instance_info = self.training_instances[training_job_name]
        
        try:
            # Ensure SSH is reachable
            self._wait_for_ssh(instance_info['public_ip'])
            
            # Create SSH client
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            key_path = Path(f"~/.ssh/{self.config.ec2_config.key_pair_name}.pem").expanduser()
            
            # Load private key
            try:
                from paramiko import RSAKey, Ed25519Key, ECDSAKey
                # Try different key types
                private_key = None
                for key_class in [RSAKey, Ed25519Key, ECDSAKey]:
                    try:
                        private_key = key_class.from_private_key_file(str(key_path))
                        break
                    except Exception:
                        continue
                
                if private_key is None:
                    raise Exception(f"Could not load private key from {key_path}")
                
                ssh_client.connect(
                    hostname=instance_info['public_ip'],
                    username='ubuntu',  # Ubuntu AMI uses ubuntu
                    pkey=private_key
                )
            except Exception as e:
                # Fallback to key_filename method
                self.logger.warning(f"Failed to load key manually: {e}, trying key_filename method")
                ssh_client.connect(
                    hostname=instance_info['public_ip'],
                    username='ubuntu',  # Ubuntu AMI uses ubuntu
                    key_filename=str(key_path)
                )
            
            # Set environment variables
            env_vars = self.config.get_training_environment_vars()
            env_setup = " && ".join([f"export {k}={v}" for k, v in env_vars.items()])
            
            # Determine S3 transcripts prefix; default to project's training transcripts path
            s3_transcripts_prefix = s3_transcripts_prefix or 'data/training/transcripts/'
            
            # Start training command using the proper training pipeline
            training_script = f"""
            cd /home/ubuntu/stranger-things-nlp
            
            # Create and activate virtual environment
            python3 -m venv venv
            source venv/bin/activate
            
            # Upgrade pip and install requirements
            pip install --upgrade pip
            # Install CUDA-enabled PyTorch for GPU instances
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            pip install boto3 transformers datasets accelerate peft bitsandbytes trl huggingface_hub
            
            # Load environment variables and authenticate with HuggingFace
            source .env
            echo "Authenticating with HuggingFace..."
            python -c "from huggingface_hub import login; import os; login(os.getenv('huggingface_token'))"
            
            {env_setup}
            
            # Verify GPU is available
            python -c "import torch; print(f'CUDA Available: {{torch.cuda.is_available()}}'); print(f'GPU Count: {{torch.cuda.device_count()}}'); print(f'Current Device: {{torch.cuda.current_device() if torch.cuda.is_available() else \'CPU\'}}'); print(f'GPU Name: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'N/A\'}}')"
            
            # Use the proper training pipeline; fetch transcripts from S3
            python training_pipeline.py --model-type {model_type} --s3-transcripts {s3_transcripts_prefix}
            
            # Log completion
            echo "Training pipeline completed at $(date)" >> training.log
            """
            
            # Execute training in background
            command = f"nohup bash -c '{training_script}' > training.log 2>&1 &"
            ssh_client.exec_command(command)
            
            # Update status
            self.training_instances[training_job_name]['status'] = 'training'
            
            self.logger.info(f"Started training for job: {training_job_name}")
            
            if stream_logs:
                # Stream logs live
                tail_cmd = f"tail -n {log_lines} -F /home/ubuntu/stranger-things-nlp/training.log"
                transport = ssh_client.get_transport()
                channel = transport.open_session()
                channel.exec_command(tail_cmd)
                try:
                    while True:
                        if channel.recv_ready():
                            data = channel.recv(4096).decode('utf-8', errors='replace')
                            print(data, end='')
                        if channel.recv_stderr_ready():
                            err = channel.recv_stderr(4096).decode('utf-8', errors='replace')
                            print(err, end='')
                        if channel.exit_status_ready():
                            break
                        time.sleep(0.2)
                except KeyboardInterrupt:
                    self.logger.info("Log streaming interrupted by user")
                finally:
                    try:
                        channel.close()
                    except Exception:
                        pass
                    ssh_client.close()
            else:
                ssh_client.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting training: {e}")
            return False
    
    def start_gradio_app(self) -> bool:
        """Start Gradio app on EC2 instance"""
        if 'main' not in self.gradio_instances:
            self.logger.error("No Gradio instance found")
            return False
        
        instance_info = self.gradio_instances['main']
        
        try:
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            key_path = Path(f"~/.ssh/{self.config.ec2_config.key_pair_name}.pem").expanduser()
            
            # Load private key
            try:
                from paramiko import RSAKey, Ed25519Key, ECDSAKey
                # Try different key types
                private_key = None
                for key_class in [RSAKey, Ed25519Key, ECDSAKey]:
                    try:
                        private_key = key_class.from_private_key_file(str(key_path))
                        break
                    except Exception:
                        continue
                
                if private_key is None:
                    raise Exception(f"Could not load private key from {key_path}")
                
                ssh_client.connect(
                    hostname=instance_info['public_ip'],
                    username='ubuntu',  # Ubuntu AMI uses ubuntu
                    pkey=private_key
                )
            except Exception as e:
                # Fallback to key_filename method
                self.logger.warning(f"Failed to load key manually: {e}, trying key_filename method")
                ssh_client.connect(
                    hostname=instance_info['public_ip'],
                    username='ubuntu',  # Ubuntu AMI uses ubuntu
                    key_filename=str(key_path)
                )
            
            # Start Gradio app
            gradio_command = f"""
            cd /home/ubuntu/stranger-things-nlp
            source venv/bin/activate
            
            export GRADIO_SERVER_NAME=0.0.0.0
            export GRADIO_SERVER_PORT={self.config.gradio_config.port}
            
            nohup python gradio_app.py > gradio.log 2>&1 &
            """
            
            ssh_client.exec_command(gradio_command)
            
            # Update status
            self.gradio_instances['main']['status'] = 'running'
            
            ssh_client.close()
            self.logger.info(f"Started Gradio app: {instance_info['gradio_url']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting Gradio app: {e}")
            return False
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate EC2 instance"""
        try:
            self.ec2_client.terminate_instances(InstanceIds=[instance_id])
            self.logger.info(f"Terminated instance: {instance_id}")
            return True
        except ClientError as e:
            self.logger.error(f"Error terminating instance: {e}")
            return False
    
    def get_training_status(self, training_job_name: str) -> Dict:
        """Get status of training job"""
        if training_job_name not in self.training_instances:
            return {'status': 'not_found'}
        
        return self.training_instances[training_job_name]
    
    def fetch_training_log(self, training_job_name: Optional[str] = None, model_type: Optional[str] = None, source: str = 'ssh', lines: int = 200) -> str:
        """Fetch recent training.log content either via SSH or from S3.
        If source='ssh', requires a running instance accessible via SSH.
        If source='s3', attempts to find latest training.log in logs/{model_type}/.
        Returns the log content as a string (may be partial)."""
        try:
            if source == 'ssh':
                # If training_job_name specified and tracked, use its IP
                ip = None
                if training_job_name and training_job_name in self.training_instances:
                    ip = self.training_instances[training_job_name].get('public_ip')
                if not ip:
                    # Fallback: find any running training instance and use its IP
                    running = [i for i in self._find_active_training_instances() if i.get('public_ip')]
                    if not running:
                        return "No running training instances with public IP found for SSH log fetch."
                    ip = running[0]['public_ip']
                # Connect and tail last N lines (not following)
                self._wait_for_ssh(ip)
                ssh_client = paramiko.SSHClient()
                ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                key_path = Path(f"~/.ssh/{self.config.ec2_config.key_pair_name}.pem").expanduser()
                try:
                    from paramiko import RSAKey, Ed25519Key, ECDSAKey
                    private_key = None
                    for key_class in [RSAKey, Ed25519Key, ECDSAKey]:
                        try:
                            private_key = key_class.from_private_key_file(str(key_path))
                            break
                        except Exception:
                            continue
                    if private_key is None:
                        raise Exception(f"Could not load private key from {key_path}")
                    ssh_client.connect(hostname=ip, username='ubuntu', pkey=private_key)
                except Exception:
                    ssh_client.connect(hostname=ip, username='ubuntu', key_filename=str(key_path))
                stdin, stdout, stderr = ssh_client.exec_command(f"tail -n {lines} /home/ubuntu/stranger-things-nlp/training.log || echo 'training.log not found'" )
                out = stdout.read().decode('utf-8', errors='replace')
                err = stderr.read().decode('utf-8', errors='replace')
                ssh_client.close()
                return out if out.strip() else err
            else:
                # S3 mode: attempt to find latest training.log under logs/{model_type}/
                if not model_type:
                    # Try to infer from active instances
                    actives = self._find_active_training_instances()
                    if actives and actives[0].get('model_type'):
                        model_type = actives[0]['model_type']
                if not model_type:
                    return "Model type not specified and could not infer for S3 log fetch."
                prefix = f"logs/{model_type}/"
                objs = self.s3_manager.list_objects(prefix)
                if not objs:
                    return f"No logs found in s3://{self.s3_manager.bucket_name}/{prefix}"
                # Find latest training.log by LastModified
                latest_key = None
                latest_time = None
                for obj in objs:
                    key = obj['Key']
                    if key.endswith('training.log'):
                        lm = obj.get('LastModified')
                        if latest_time is None or (lm and lm > latest_time):
                            latest_time = lm
                            latest_key = key
                if not latest_key:
                    return f"No training.log found under s3://{self.s3_manager.bucket_name}/{prefix}"
                # Download to temp and read last N lines
                import tempfile
                import io
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.close()
                ok = self.s3_manager.download_file(latest_key, tmp.name, show_progress=False)
                if not ok:
                    return f"Failed to download s3://{self.s3_manager.bucket_name}/{latest_key}"
                with open(tmp.name, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.readlines()
                return ''.join(content[-lines:])
        except Exception as e:
            return f"Error fetching training log: {e}"

    def list_instances(self) -> Dict:
        """List all project instances by querying AWS (project-tagged)."""
        # Training instances (pending/running/stopping/stopped)
        training_filters = [
            {'Name': 'tag:Project', 'Values': ['stranger-things-nlp']},
            {'Name': 'tag:Purpose', 'Values': ['training']},
            {'Name': 'instance-state-name', 'Values': ['pending', 'running', 'stopping', 'stopped']},
        ]
        training_resp = self.ec2_client.describe_instances(Filters=training_filters)
        training: Dict[str, Dict] = {}
        for r in training_resp.get('Reservations', []):
            for inst in r.get('Instances', []):
                state = inst['State']['Name']
                public_ip = inst.get('PublicIpAddress', '')
                private_ip = inst.get('PrivateIpAddress', '')
                tags = inst.get('Tags', [])
                job_name = self._get_tag(tags, 'JobName') or inst['InstanceId']
                model_type = self._get_tag(tags, 'ModelType') or self._infer_model_type_from_name(self._get_tag(tags, 'Name'))
                training[job_name] = {
                    'instance_id': inst['InstanceId'],
                    'model_type': model_type,
                    'public_ip': public_ip,
                    'private_ip': private_ip,
                    'status': state,
                }
        
        # Gradio instances
        gradio_filters = [
            {'Name': 'tag:Project', 'Values': ['stranger-things-nlp']},
            {'Name': 'tag:Purpose', 'Values': ['gradio-hosting']},
            {'Name': 'instance-state-name', 'Values': ['pending', 'running', 'stopping', 'stopped']},
        ]
        gradio_resp = self.ec2_client.describe_instances(Filters=gradio_filters)
        gradio: Dict[str, Dict] = {}
        for r in gradio_resp.get('Reservations', []):
            for inst in r.get('Instances', []):
                state = inst['State']['Name']
                public_ip = inst.get('PublicIpAddress', '')
                private_ip = inst.get('PrivateIpAddress', '')
                name = next((t['Value'] for t in inst.get('Tags', []) if t.get('Key') == 'Name'), inst['InstanceId'])
                gradio_url = f"http://{public_ip}:{self.config.gradio_config.port}" if public_ip else ''
                gradio[name] = {
                    'instance_id': inst['InstanceId'],
                    'public_ip': public_ip,
                    'private_ip': private_ip,
                    'status': state,
                    'gradio_url': gradio_url,
                }
        
        return {
            'training': training,
            'gradio': gradio,
        }
    
    def _wait_for_spot_instance(self, spot_request_id: str, timeout: int = 300) -> str:
        """Wait for spot instance request to be fulfilled"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.ec2_client.describe_spot_instance_requests(
                SpotInstanceRequestIds=[spot_request_id]
            )
            
            request = response['SpotInstanceRequests'][0]
            state = request['State']
            
            if state == 'active':
                return request['InstanceId']
            elif state == 'failed':
                raise Exception(f"Spot request failed: {request.get('Fault', 'Unknown error')}")
            
            time.sleep(10)
        
        raise Exception("Timeout waiting for spot instance")
    
    def _get_instance_info(self, instance_id: str) -> Dict:
        """Get instance information"""
        response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        
        return {
            'instance_id': instance_id,
            'public_ip': instance.get('PublicIpAddress', ''),
            'private_ip': instance.get('PrivateIpAddress', ''),
            'state': instance['State']['Name']
        }

    def _get_tag(self, tags: List[Dict], key: str) -> Optional[str]:
        if not tags:
            return None
        for t in tags:
            if t.get('Key') == key:
                return t.get('Value')
        return None

    def _infer_model_type_from_name(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        # Expected pattern: stranger-things-training-<model_type>
        parts = name.split('-')
        if len(parts) >= 4 and parts[0] == 'stranger' and parts[1] == 'things' and parts[2] == 'training':
            return parts[3]
        return None

    def _find_active_training_instances(self) -> List[Dict]:
        """Find active (pending/running) training instances for this project."""
        filters = [
            {'Name': 'tag:Project', 'Values': ['stranger-things-nlp']},
            {'Name': 'tag:Purpose', 'Values': ['training']},
            {'Name': 'instance-state-name', 'Values': ['pending', 'running']},
        ]
        resp = self.ec2_client.describe_instances(Filters=filters)
        results: List[Dict] = []
        for r in resp.get('Reservations', []):
            for inst in r.get('Instances', []):
                tags = inst.get('Tags', [])
                results.append({
                    'instance_id': inst['InstanceId'],
                    'public_ip': inst.get('PublicIpAddress', ''),
                    'private_ip': inst.get('PrivateIpAddress', ''),
                    'state': inst['State']['Name'],
                    'model_type': self._get_tag(tags, 'ModelType') or self._infer_model_type_from_name(self._get_tag(tags, 'Name'))
                })
        return results

    def _wait_for_public_ip(self, instance_id: str, timeout: int = 300, interval: int = 5) -> Dict:
        """Wait until the instance has a Public IP assigned and return instance info."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            info = self._get_instance_info(instance_id)
            if info.get('public_ip'):
                return info
            time.sleep(interval)
        raise TimeoutError(f"Timed out waiting for Public IP on instance {instance_id}")

    def _wait_for_instance_status_ok(self, instance_id: str, timeout: int = 600, delay: int = 10):
        """Wait for EC2 instance and system status checks to pass (Status=ok)."""
        self.logger.info("Waiting for instance status checks to pass (Status=ok)...")
        waiter = self.ec2_client.get_waiter('instance_status_ok')
        max_attempts = max(1, timeout // delay)
        waiter.wait(InstanceIds=[instance_id], WaiterConfig={'Delay': delay, 'MaxAttempts': max_attempts})

    def _wait_for_ssh(self, host: str, port: int = 22, timeout: int = 600, interval: int = 5):
        """Wait until TCP port 22 on host is reachable."""
        self.logger.info(f"Waiting for SSH to be reachable at {host}:{port}...")
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(interval)
            try:
                sock.connect((host, port))
                sock.shutdown(socket.SHUT_RDWR)
                return True
            except Exception as e:
                last_err = e
                time.sleep(interval)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        raise TimeoutError(f"Timed out waiting for SSH on {host}:{port}: {last_err}")
    
    def _get_gradio_user_data_script(self) -> str:
        """Get user data script for Gradio instance"""
        base_script = self.config.get_ec2_user_data_script()
        
        # Add Gradio-specific setup
        gradio_setup = """
# Additional setup for Gradio hosting
sudo -u ubuntu bash << 'EOF'
cd /home/ubuntu/stranger-things-nlp

# Install nginx for reverse proxy (optional)
# sudo yum install -y nginx  # Use yum for Amazon Linux

echo "Gradio instance setup completed" >> gradio_setup.log
EOF"""
        
        return base_script + "\n" + gradio_setup
    
    def _upload_directory_recursive(self, sftp, local_dir: str, remote_dir: str):
        """Upload directory recursively via SFTP"""
        local_path = Path(local_dir)
        
        # Create remote directory
        try:
            sftp.mkdir(remote_dir)
        except:
            pass  # Directory might already exist
        
        for item in local_path.iterdir():
            local_item_path = str(item)
            remote_item_path = f"{remote_dir}/{item.name}"
            
            if item.is_file():
                sftp.put(local_item_path, remote_item_path)
            elif item.is_dir() and not item.name.startswith('.'):
                self._upload_directory_recursive(sftp, local_item_path, remote_item_path)