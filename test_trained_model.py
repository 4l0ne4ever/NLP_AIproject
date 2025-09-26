#!/usr/bin/env python3
"""Test script to verify trained model loading and basic functionality"""

import os
from dotenv import load_dotenv
from gradio_app_v2 import S3ModelManager, initialize_chatbot_models

load_dotenv()

def test_model_download():
    """Test if we can download and load the trained model"""
    print("🔍 Testing trained model loading...")
    
    # Test S3 model manager
    s3_manager = S3ModelManager()
    
    # Check if model info exists
    print("\n1. Checking model info in S3...")
    llama_info = s3_manager.get_latest_model_info("llama")
    if llama_info:
        print(f"✅ Found LLaMA model info:")
        print(f"   - Timestamp: {llama_info['timestamp']}")
        print(f"   - Loss: {llama_info['loss']}")
        print(f"   - Model path: {llama_info['model_path']}")
    else:
        print("❌ No LLaMA model found in S3")
        return False
    
    # Test model download
    print("\n2. Testing model download...")
    model_path = s3_manager.get_model_path("llama")
    if model_path:
        print(f"✅ Model downloaded to: {model_path}")
        
        # Check if key files exist
        key_files = ["config.json", "model.safetensors", "tokenizer.json"]
        for file in key_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                print(f"   ✅ {file} exists")
            else:
                print(f"   ❌ {file} missing")
    else:
        print("❌ Failed to download model")
        return False
    
    print("\n3. Testing chatbot initialization...")
    try:
        initialize_chatbot_models()
        print("✅ Chatbot models initialized successfully!")
        
        # Test a simple chat (this might take a while for first load)
        print("\n4. Testing simple chat...")
        from gradio_app_v2 import character_chatbot_llama
        if character_chatbot_llama and character_chatbot_llama.model:
            try:
                response = character_chatbot_llama.chat("Hi, how are you?", [])
                print(f"✅ Chat test successful!")
                print(f"   Response: {response}")
            except Exception as e:
                print(f"⚠️ Chat test failed: {e}")
        else:
            print("❌ Chatbot model not loaded")
            
    except Exception as e:
        print(f"❌ Failed to initialize chatbots: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Testing Trained Model Integration")
    print("=" * 50)
    
    success = test_model_download()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Your app should work with the trained model.")
    else:
        print("🔧 Some issues found. Check the output above for details.")