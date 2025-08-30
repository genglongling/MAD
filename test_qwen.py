#!/usr/bin/env python3
"""
Simple test script for Qwen model
"""

import os
import sys
sys.path.append('.')

from src.debate.models import LocalModel

def test_qwen():
    """Test Qwen model generation"""
    try:
        print("Testing Qwen model...")
        
        # Create model instance
        model = LocalModel("Qwen/Qwen2.5-7B-Instruct", temperature=0.3, max_tokens=100)
        
        # Test simple prompt
        messages = [
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ]
        
        print("Generating response...")
        response = model.invoke(messages)
        
        print(f"Response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"Error testing Qwen: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_qwen()
