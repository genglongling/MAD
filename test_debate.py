#!/usr/bin/env python3
"""
Test script for Qwen model with debate prompt
"""

import os
import sys
sys.path.append('.')

from src.debate.models import LocalModel

def test_debate_prompt():
    """Test Qwen model with debate prompt"""
    try:
        print("Testing Qwen model with debate prompt...")
        
        # Create model instance
        model = LocalModel("Qwen/Qwen2.5-7B-Instruct", temperature=0.7, max_tokens=100)
        
        # Test debate prompt
        messages = [
            {"role": "user", "content": "You are presented with the following multiple-choice question:\nQuestion: As water starts to freeze, the molecules of water\nChoices: A) gain thermal energy, B) move more freely, C) increase in size, D) decrease in speed.\n\nOutput (strict JSON):\n{\"output\": {\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}, \"reason\": {\"A\": rA, \"B\": rB, \"C\": rC, \"D\": rD}}"}
        ]
        
        print("Generating debate response...")
        response = model.invoke(messages)
        
        print(f"Response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"Error testing debate prompt: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_debate_prompt()
