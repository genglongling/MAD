#!/usr/bin/env python3
"""
Setup script for Qwen2.5-7B-Instruct model
This script downloads the model and tokenizer to local cache
"""

import os
import sys
from pathlib import Path

def setup_qwen():
    """Download and setup Qwen2.5-7B-Instruct model"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Setting up Qwen2.5-7B-Instruct model...")
        print("This will download ~14GB of model files to your HuggingFace cache.")
        print("Make sure you have sufficient disk space and a stable internet connection.")
        
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
        
        print("\n✅ Qwen2.5-7B-Instruct model setup complete!")
        print("The model is now cached and ready to use with the debate system.")
        print("\nTo use it, make sure your configs/benchmark.yaml includes one of these pairings:")
        print("  - gpt5_qwen25")
        print("  - gpt4_qwen25")
        print("  - claude4_qwen25")
        print("  - gemini25pro_qwen25")
        print("  - qwen25_qwen25")
        
    except ImportError:
        print("❌ Error: transformers library not found.")
        print("Please install it first:")
        print("pip install transformers torch accelerate")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error setting up Qwen model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_qwen()
