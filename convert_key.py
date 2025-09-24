#!/usr/bin/env python3
"""
Convert RSA private key to OpenSSH format for compatibility with paramiko
"""

import os
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

def convert_rsa_to_openssh(rsa_key_path, output_path):
    """Convert RSA private key to OpenSSH format"""
    
    with open(rsa_key_path, 'rb') as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend()
        )
    
    # Serialize to PEM format (which is compatible with paramiko)
    openssh_key = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    with open(output_path, 'wb') as f:
        f.write(openssh_key)
    
    # Set appropriate permissions
    os.chmod(output_path, 0o600)
    
    print(f"Converted {rsa_key_path} to OpenSSH format at {output_path}")

if __name__ == "__main__":
    # Convert the stranger-things-key-dct.pem
    rsa_key = Path("~/.ssh/stranger-things-key-dct.pem").expanduser()
    openssh_key = Path("~/.ssh/stranger-things-key-dct-openssh.pem").expanduser()
    
    if rsa_key.exists():
        convert_rsa_to_openssh(rsa_key, openssh_key)
    else:
        print(f"RSA key not found at {rsa_key}")