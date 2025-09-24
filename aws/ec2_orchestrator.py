import boto3
import time
import json
import logging
import paramiko
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
                                {'Key': 'Purpose', 'Value': 'training'}
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
            
            # Get instance details
            instance_info = self._get_instance_info(instance_id)
            
            # Store training job info
            self.training_instances[training_job_name] = {
                'instance_id': instance_id,
                'model_type': model_type,
                'public_ip': instance_info['public_ip'],
                'private_ip': instance_info['private_ip'],
                'launch_time': time.time(),
                'status': 'initializing'
            }
            
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
            
            instance_info = self._get_instance_info(instance_id)
            
            self.gradio_instances['main'] = {
                'instance_id': instance_id,
                'public_ip': instance_info['public_ip'],
                'private_ip': instance_info['private_ip'],
                'launch_time': time.time(),
                'status': 'initializing',
                'gradio_url': f"http://{instance_info['public_ip']}:{self.config.gradio_config.port}"
            }
            
            self.logger.info(f"Gradio instance ready: {instance_info['public_ip']}")
            return self.gradio_instances['main']
            
        except ClientError as e:
            self.logger.error(f"Error launching Gradio instance: {e}")
            raise
    
    def deploy_code_to_instance(self, instance_info: Dict, 
                               local_project_path: str = ".") -> bool:
        """Deploy code to EC2 instance via SCP"""
        try:
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
                    username='ec2-user',  # Amazon Linux 2 uses ec2-user
                    pkey=private_key
                )
            except Exception as e:
                # Fallback to key_filename method
                self.logger.warning(f"Failed to load key manually: {e}, trying key_filename method")
                ssh_client.connect(
                    hostname=instance_info['public_ip'],
                    username='ec2-user',  # Amazon Linux 2 uses ec2-user
                    key_filename=str(key_path)
                )
            
            # Create base directory first
            ssh_client.exec_command(f"mkdir -p /home/ec2-user/stranger-things-nlp")
            
            # Upload project files
            sftp = ssh_client.open_sftp()
            
            local_path = Path(local_project_path)
            remote_base = "/home/ec2-user/stranger-things-nlp"
            
            # Ensure remote base directory exists
            try:
                sftp.mkdir(remote_base)
            except Exception:
                pass  # Directory might already exist
            
            # Upload essential files
            essential_files = [
                'requirements.txt',
                'gradio_app.py',
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
            
            # Install requirements
            commands = [
                "cd /home/ec2-user/stranger-things-nlp",
                "python3 -m venv venv",  # Create virtual environment first
                "source venv/bin/activate",
                "pip install -r requirements.txt"
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
    
    def start_training(self, training_job_name: str, model_type: str = "llama") -> bool:
        """Start training on EC2 instance"""
        if training_job_name not in self.training_instances:
            self.logger.error(f"Training job not found: {training_job_name}")
            return False
        
        instance_info = self.training_instances[training_job_name]
        
        try:
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
                    username='ec2-user',  # Amazon Linux 2 uses ec2-user
                    pkey=private_key
                )
            except Exception as e:
                # Fallback to key_filename method
                self.logger.warning(f"Failed to load key manually: {e}, trying key_filename method")
                ssh_client.connect(
                    hostname=instance_info['public_ip'],
                    username='ec2-user',  # Amazon Linux 2 uses ec2-user
                    key_filename=str(key_path)
                )
            
            # Set environment variables
            env_vars = self.config.get_training_environment_vars()
            env_setup = " && ".join([f"export {k}={v}" for k, v in env_vars.items()])
            
            # Start training command
            training_script = f"""
            cd /home/ec2-user/stranger-things-nlp
            source venv/bin/activate
            {env_setup}
            
            python -c "
            from character_chatbot.character_chatbot import CharacterChatbot
            import os
            
            # Download training data from S3
            # This would be implemented in the updated chatbot class
            chatbot = CharacterChatbot(
                model_path='christopherxzyx/StrangerThings_{model_type.title()}-3-4B_ec2',
                data_path='/tmp/training_data/',
                huggingface_token=os.getenv('HUGGINGFACE_TOKEN')
            )
            print('Training completed successfully')
            "
            """
            
            # Execute training in background
            command = f"nohup bash -c '{training_script}' > training.log 2>&1 &"
            ssh_client.exec_command(command)
            
            # Update status
            self.training_instances[training_job_name]['status'] = 'training'
            
            ssh_client.close()
            self.logger.info(f"Started training for job: {training_job_name}")
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
                    username='ec2-user',  # Amazon Linux 2 uses ec2-user
                    pkey=private_key
                )
            except Exception as e:
                # Fallback to key_filename method
                self.logger.warning(f"Failed to load key manually: {e}, trying key_filename method")
                ssh_client.connect(
                    hostname=instance_info['public_ip'],
                    username='ec2-user',  # Amazon Linux 2 uses ec2-user
                    key_filename=str(key_path)
                )
            
            # Start Gradio app
            gradio_command = f"""
            cd /home/ec2-user/stranger-things-nlp
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
    
    def list_instances(self) -> Dict:
        """List all project instances"""
        return {
            'training': self.training_instances,
            'gradio': self.gradio_instances
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
    
    def _get_gradio_user_data_script(self) -> str:
        """Get user data script for Gradio instance"""
        base_script = self.config.get_ec2_user_data_script()
        
        # Add Gradio-specific setup
        gradio_setup = """
# Additional setup for Gradio hosting
sudo -u ec2-user bash << 'EOF'
cd /home/ec2-user/stranger-things-nlp

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