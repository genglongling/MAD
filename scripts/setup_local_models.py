#!/usr/bin/env python3
"""
Setup script for local models (Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct)
This script downloads the models and tokenizers to local cache
"""

import os
import sys
import argparse
from pathlib import Path

def setup_qwen():
    """Download and setup Qwen2.5-7B-Instruct model"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Setting up Qwen2.5-7B-Instruct model...")
        print("This will download ~14GB of model files to your HuggingFace cache.")
        
        # Model name
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        print(f"\nDownloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Downloading model for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Qwen2.5-7B-Instruct model setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Qwen model: {e}")
        return False

def setup_llama():
    """Download and setup Llama3.1-8B-Instruct model"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("\nSetting up Llama3.1-8B-Instruct model...")
        print("This will download ~16GB of model files to your HuggingFace cache.")
        print("\nNote: You may need to request access to Llama models on HuggingFace:")
        print("https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
        
        # Model name
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        print(f"\nDownloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Downloading model for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Llama3.1-8B-Instruct model setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Llama model: {e}")
        print("\nIf you get an access error, make sure to:")
        print("1. Request access to Llama models on HuggingFace")
        print("2. Login with: huggingface-cli login")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup local models for debate system")
    parser.add_argument("--qwen", action="store_true", help="Setup Qwen2.5-7B-Instruct")
    parser.add_argument("--llama", action="store_true", help="Setup Llama3.1-8B-Instruct")
    parser.add_argument("--all", action="store_true", help="Setup both models")
    
    args = parser.parse_args()
    
    if not any([args.qwen, args.llama, args.all]):
        print("Please specify which model(s) to setup:")
        print("  --qwen    Setup Qwen2.5-7B-Instruct")
        print("  --llama   Setup Llama3.1-8B-Instruct")
        print("  --all     Setup both models")
        sys.exit(1)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("❌ Error: transformers library not found.")
        print("Please install it first:")
        print("pip install transformers torch accelerate")
        sys.exit(1)
    
    print("Setting up local models for debate system...")
    print("Make sure you have sufficient disk space and a stable internet connection.")
    
    success_count = 0
    
    if args.qwen or args.all:
        if setup_qwen():
            success_count += 1
    
    if args.llama or args.all:
        if setup_llama():
            success_count += 1
    
    print(f"\n🎉 Setup complete! {success_count} model(s) successfully installed.")
    print("\nTo use these models, make sure your configs/benchmark.yaml includes:")
    print("  - qwen_qwen   (Qwen self-debate)")
    print("  - qwen_llama  (Qwen vs Llama)")
    print("  - llama_llama (Llama self-debate)")

if __name__ == "__main__":
    main()
