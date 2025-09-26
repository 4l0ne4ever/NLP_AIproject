#!/usr/bin/env python3
"""
Volume Diagnostic Script for EC2 Training Instances

This script helps diagnose volume-related issues on EC2 instances.
Run this script on your EC2 instance to check if volumes are properly mounted.
"""

import subprocess
import os
import sys
from pathlib import Path

def run_command(cmd):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_mounted_volumes():
    """Check if all required volumes are mounted"""
    print(" Checking mounted volumes...")
    
    required_mounts = {
        '/': 'Root filesystem',
        '/mnt/models': 'Model storage volume',
        '/mnt/cache': 'Cache volume'
    }
    
    # Get mounted filesystems
    returncode, stdout, stderr = run_command("df -h")
    
    if returncode != 0:
        print(f"Error getting filesystem info: {stderr}")
        return False
    
    mounted_paths = set()
    for line in stdout.split('\n')[1:]:  # Skip header
        if line.strip():
            parts = line.split()
            if len(parts) >= 6:
                mount_point = parts[5]
                mounted_paths.add(mount_point)
    
    print("\n Mount Point Status:")
    all_mounted = True
    for mount_point, description in required_mounts.items():
        if mount_point in mounted_paths:
            print(f" {mount_point:<15} - {description}")
        else:
            print(f" {mount_point:<15} - {description} (NOT MOUNTED)")
            all_mounted = False
    
    return all_mounted

def check_volume_space():
    """Check available space on volumes"""
    print("\n Volume Space Usage:")
    
    volumes_to_check = ['/', '/mnt/models', '/mnt/cache']
    
    for volume in volumes_to_check:
        if os.path.exists(volume):
            try:
                stat = os.statvfs(volume)
                total = stat.f_blocks * stat.f_frsize
                available = stat.f_bavail * stat.f_frsize
                used = total - available
                
                total_gb = total / (1024**3)
                available_gb = available / (1024**3)
                used_gb = used / (1024**3)
                used_percent = (used / total) * 100
                
                print(f" {volume:<15}: {used_gb:.1f}GB used / {total_gb:.1f}GB total ({used_percent:.1f}% used)")
                
                # Warn if usage is high
                if used_percent > 90:
                    print(f"    WARNING: Volume {volume} is {used_percent:.1f}% full!")
                elif used_percent > 80:
                    print(f"     Volume {volume} is getting full ({used_percent:.1f}%)")
                
            except Exception as e:
                print(f" {volume:<15}: Error checking space - {e}")
        else:
            print(f" {volume:<15}: Path does not exist")

def check_block_devices():
    """Check available block devices"""
    print("\n Block Devices:")
    
    returncode, stdout, stderr = run_command("lsblk")
    if returncode == 0:
        print(stdout)
    else:
        print(f"Error listing block devices: {stderr}")

def check_directory_structure():
    """Check if required directories exist and have correct permissions"""
    print("\nDirectory Structure Check:")
    
    required_dirs = [
        '/mnt/models',
        '/mnt/models/checkpoints',
        '/mnt/models/trained_models', 
        '/mnt/models/datasets',
        '/mnt/cache',
        '/mnt/cache/transformers',
        '/mnt/cache/huggingface',
        '/home/ec2-user/stranger-things-nlp'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            # Check ownership and permissions
            stat_info = path.stat()
            permissions = oct(stat_info.st_mode)[-3:]
            
            # Get owner info
            try:
                import pwd
                owner = pwd.getpwuid(stat_info.st_uid).pw_name
            except:
                owner = f"uid:{stat_info.st_uid}"
                
            print(f"  {dir_path:<35} (owner: {owner}, perms: {permissions})")
        else:
            print(f"  {dir_path:<35} (MISSING)")

def check_environment_variables():
    """Check if training environment variables are set correctly"""
    print("\n Training Environment Variables:")
    
    expected_vars = {
        'TRANSFORMERS_CACHE': '/mnt/cache/transformers',
        'HF_HOME': '/mnt/cache/huggingface', 
        'MODEL_STORAGE_PATH': '/mnt/models',
        'CHECKPOINT_PATH': '/mnt/models/checkpoints',
        'CUDA_VISIBLE_DEVICES': '0'
    }
    
    for var_name, expected_value in expected_vars.items():
        actual_value = os.getenv(var_name)
        if actual_value:
            if actual_value == expected_value:
                print(f"  {var_name:<20} = {actual_value}")
            else:
                print(f"   {var_name:<20} = {actual_value} (expected: {expected_value})")
        else:
            print(f" {var_name:<20} = NOT SET (expected: {expected_value})")

def check_gpu():
    """Check GPU availability"""
    print("\n GPU Check:")
    
    # Check if nvidia-smi is available
    returncode, stdout, stderr = run_command("nvidia-smi")
    if returncode == 0:
        print("  NVIDIA GPU detected:")
        # Show just the header and first GPU info
        lines = stdout.split('\n')
        for i, line in enumerate(lines):
            if i < 15:  # Show first 15 lines (header + GPU info)
                print(f"    {line}")
            elif "MiB |" in line:  # Show memory usage line
                print(f"    {line}")
                break
    else:
        print("  No NVIDIA GPU detected or nvidia-smi not available")
        print(f"     Error: {stderr}")

def main():
    """Main diagnostic function"""
    print(" EC2 Training Instance Volume Diagnostic")
    print("=" * 50)
    
    # Run all checks
    volumes_ok = check_mounted_volumes()
    check_volume_space()
    check_block_devices()
    check_directory_structure()
    check_environment_variables()
    check_gpu()
    
    print("\n" + "=" * 50)
    if volumes_ok:
        print(" Volume configuration looks good!")
        print("   You should be able to proceed with training.")
    else:
        print("Volume configuration issues detected!")
        print("   Please fix the missing volumes before training.")
        
    print("\n Tips:")
    print("   • Make sure additional EBS volumes are attached to the instance")
    print("   • Check that the user-data script completed successfully")
    print("   • View user-data logs: sudo cat /var/log/user-data.log")
    print("   • Manual mount: sudo mount /dev/sdf /mnt/models")

if __name__ == "__main__":
    main()