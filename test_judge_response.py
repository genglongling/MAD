#!/usr/bin/env python3
"""
Test script to verify judge model response
"""

import os
import sys
sys.path.append('.')

from src.debate.models import LocalModel
from src.debate.prompts import parse_judge_json

def test_judge_response():
    """Test judge model with the exact judge prompt"""
    try:
        print("Testing judge model...")
        
        # Create judge model instance
        judge_model = LocalModel("Qwen/Qwen2.5-7B-Instruct", temperature=0.2, max_tokens=2048)
        
        # Test the exact judge prompt
        system_prompt = """You are a rigorous, deterministic judge. Apply the CRIT algorithm directly and output STRICT JSON only.
Output schema:
{
  "outputA": {"A": pA, "B": pB, "C": pC, "D": pD},
  "outputB": {"A": pA, "B": pB, "C": pC, "D": pD},
  "CRIT_A": float,
  "CRIT_B": float,
  "NOTE_A": str,
  "NOTE_B": str
}
Rules:
- No prose outside JSON; adhere exactly to the schema above.
- Each output distribution is nonnegative and sums to 1.
- CRIT_A / CRIT_B ∈ [0,1], computed per CRIT steps below.
- NOTE_A / NOTE_B must concisely explain how γ (strength) and θ (reliability) were derived and aggregated from reasons and rival reasons, including any evidence used."""
        
        user_prompt = """Round: 1
Question: As water starts to freeze, the molecules of water
Choices: A) gain thermal energy, B) move more freely, C) increase in size, D) decrease in speed.

# Agent A (Round 1)
outputA: {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.7}
reasonA: {"A": "Water molecules losing energy is a direct contradiction of freezing. Freezing implies a loss of thermal energy, not gaining.", "B": "Molecules move less freely as water freezes, not more. Movement is restricted.", "C": "Water molecules do not increase in size during freezing, a critical scientific fact often misunderstood.", "D": "Water molecules indeed decrease in speed as they form a crystalline structure, the correct and polarizing answer."}

# Agent B (Round 1)
outputB: {"A": 0.05, "B": 0.05, "C": 0.1, "D": 0.8}
reasonB: {"A": "Freezing water cannot gain thermal energy as it is releasing it to its surroundings.", "B": "Water molecules slow down, not move more freely, as it freezes.", "C": "Water expands when freezing, but this is not a direct consequence of decreased speed.", "D": "Water molecules decrease in speed as they form a crystalline structure during freezing."}

CRIT: "Function Γ = CRIT(d)
Input: document d   Output: validation score Γ
Vars: Ω claim; R and R′ sets of reasons and rival reasons
Subs: CLAIM(), FINDDOC(), VALIDATE()
Begin
#1–#2 Identify in d the claim Ω. Find a set of supporting reasons R for Ω.
#3 For each r ∈ R evaluate r ⇒ Ω.
   If CLAIM(r) then (γ_r, θ_r) = CRIT(FINDDOC(r)).
   Else (γ_r, θ_r) = VALIDATE(r ⇒ Ω).
#4–#6–#7–#8 Find a set of rival reasons R′ against Ω.
   #5 For each r′ ∈ R′ compute (γ_{r′}, θ_{r′}) = VALIDATE(r′ ⇒ Ω).
   Compute a weighted sum Γ from {γ_r, θ_r, γ_{r′}, θ_{r′}}.
   Analyze arguments to justify the final Γ score.
   Reflect on transfer of CRIT to other contexts.
End"

Output STRICT JSON only:
{
  "outputA": {"A": pA, "B": pB, "C": pC, "D": pD},
  "outputB": {"A": pA, "B": pB, "C": pC, "D": pD},
  "CRIT_A": float,
  "CRIT_B": float,
  "NOTE_A": "string",
  "NOTE_B": "string"
}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print("Generating judge response...")
        response = judge_model.invoke(messages)
        
        print(f"Raw judge response: {repr(response.content)}")
        print(f"Response length: {len(response.content)}")
        
        # Test parsing
        choice_keys = ["A", "B", "C", "D"]
        parsed = parse_judge_json(response.content, choice_keys)
        
        print(f"Parsed judge result: {parsed}")
        
        return True
        
    except Exception as e:
        print(f"Error testing judge response: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_judge_response()
