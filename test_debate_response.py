#!/usr/bin/env python3
"""
Test script to see the exact raw response from Qwen model with debate prompt
"""

import os
import sys
sys.path.append('.')

from src.debate.models import LocalModel
from src.debate.prompts import parse_json_or_fallback

def test_debate_response():
    """Test Qwen model with the exact debate prompt"""
    try:
        print("Testing Qwen model with debate prompt...")
        
        # Create model instance
        model = LocalModel("Qwen/Qwen2.5-7B-Instruct", temperature=0.7, max_tokens=100)
        
        # Test the exact debate prompt
        system_prompt = """You are a careful multiple-choice reasoner. Always answer in STRICT JSON and nothing else.
Schema:
{
  "output": {"A": pA, "B": pB, "C": pC, "D": pD},
  "reason": {"A": rA, "B": rB, "C": rC, "D": rD}
}
Constraints:
- pA..pD are nonnegative and sum to 1 (normalized).
- Each rX is a short argumentative rationale for that option.
- Do not add any keys, prose, or explanations outside the JSON."""
        
        user_prompt = """You are presented with the following multiple-choice question:
Question: As water starts to freeze, the molecules of water
Choices: A) gain thermal energy, B) move more freely, C) increase in size, D) decrease in speed.

Tone: Highly confrontational; raise strong objections.
Emphasis: Highlight risks, downsides, unintended consequences, inequities.
Language: Definitive and polarizing.

Output (strict JSON):
{"output": {"A": pA, "B": pB, "C": pC, "D": pD}, "reason": {"A": rA, "B": rB, "C": rC, "D": rD}}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print("Generating debate response...")
        response = model.invoke(messages)
        
        print(f"Raw response: {repr(response.content)}")
        print(f"Response length: {len(response.content)}")
        
        # Test parsing
        choice_keys = ["A", "B", "C", "D"]
        parsed = parse_json_or_fallback(response.content, choice_keys)
        
        print(f"Parsed result: {parsed}")
        
        return True
        
    except Exception as e:
        print(f"Error testing debate response: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_debate_response()
