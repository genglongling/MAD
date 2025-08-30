#!/usr/bin/env python3
"""
Setup script for Llama3.1-8B-Instruct model
This script downloads the model and tokenizer to local cache
"""

import os
import sys
from pathlib import Path

def setup_llama():
    """Download and setup Llama3.1-8B-Instruct model"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Setting up Llama3.1-8B-Instruct model...")
        print("This will download ~16GB of model files to your HuggingFace cache.")
        print("Make sure you have sufficient disk space and a stable internet connection.")
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
        
        print("\n✅ Llama3.1-8B-Instruct model setup complete!")
        print("The model is now cached and ready to use with the debate system.")
        print("\nTo use it, make sure your configs/benchmark.yaml includes one of these pairings:")
        print("  - qwen_qwen")
        print("  - qwen_llama")
        print("  - llama_llama")
        
    except ImportError:
        print("❌ Error: transformers library not found.")
        print("Please install it first:")
        print("pip install transformers torch accelerate")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error setting up Llama model: {e}")
        print("\nIf you get an access error, make sure to:")
        print("1. Request access to Llama models on HuggingFace")
        print("2. Login with: huggingface-cli login")
        sys.exit(1)

if __name__ == "__main__":
    setup_llama()
