"""
QASC (allenai/qasc) multiple-choice 8 options originally. We'll map to A..D by taking the first 4 distractors+answer if needed.
"""
from typing import Iterator, Dict
from datasets import load_dataset

LETTER = ["A","B","C","D"]

def iter_items(split: str="validation") -> Iterator[Dict]:
    ds = load_dataset("allenai/qasc", split=split)
    for ex in ds:
        # qasc has 'choices' keys: 'text' list and 'label' list like ['A','B',...]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        # if >4, keep first 4 (including correct label's position if within)
        pairs = list(zip(labels, texts))[:4]
        choices = {lab: txt for lab, txt in pairs}
        ans = ex["answerKey"]
        # If the answer label not in our truncated set, default to first label
        answer = ans if ans in choices else list(choices.keys())[0]
        yield {
            "question": ex["question"],
            "choices": choices,
            "answer": answer
        }
