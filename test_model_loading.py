#!/usr/bin/env python3
"""
Test script to verify local model loading
"""

import os
import sys
sys.path.append('.')

from src.debate.models import LLMFactory

def test_model_loading():
    """Test that local models are loaded correctly"""
    try:
        print("Testing local model loading...")
        
        # Test Qwen model loading
        print("Loading Qwen model...")
        qwen_model = LLMFactory.make("local", "Qwen/Qwen2.5-7B-Instruct", temperature=0.7, max_tokens=100)
        print(f"Qwen model type: {type(qwen_model)}")
        print(f"Qwen model provider: {qwen_model.provider}")
        print(f"Qwen model name: {qwen_model.model}")
        
        # Test a simple response
        messages = [
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ]
        
        print("Testing Qwen response...")
        response = qwen_model.invoke(messages)
        print(f"Qwen response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"Error testing model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()
