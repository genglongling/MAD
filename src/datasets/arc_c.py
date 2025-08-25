"""
ARC-Challenge loader via Hugging Face datasets (allenai/ai2_arc).
Outputs a uniform iterator of dicts: {question, choices{A..D}, answer}
"""
from typing import Iterator, Dict
from datasets import load_dataset

LETTER = ["A","B","C","D"]

def iter_items(split: str="validation") -> Iterator[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    for ex in ds:
        # ai2_arc has 'choices': {'text': [...], 'label': ['A','B',...]}
        choices = {lab: txt for lab, txt in zip(ex["choices"]["label"], ex["choices"]["text"])}
        yield {
            "question": ex["question"],
            "choices": choices,
            "answer": ex["answerKey"]
        }
