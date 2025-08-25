"""
HellaSwag (Rowan/hellaswag). Four endings; label is 0..3
"""
from typing import Iterator, Dict
from datasets import load_dataset

LETTER = ["A","B","C","D"]

def iter_items(split: str="validation") -> Iterator[Dict]:
    ds = load_dataset("Rowan/hellaswag", split=split)
    for ex in ds:
        choices = {LETTER[i]: ex["endings"][i] for i in range(4)}
        q = ex.get("ctx_a","") + " " + ex.get("ctx_b","")
        answer = LETTER[int(ex["label"])]
        yield {
            "question": q.strip(),
            "choices": choices,
            "answer": answer
        }
