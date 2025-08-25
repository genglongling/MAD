"""
LogiQA English (lucasmccabe/logiqa). Four-choice logical reasoning.
"""
from typing import Iterator, Dict
from datasets import load_dataset

LETTER = ["A","B","C","D"]

def iter_items(split: str="validation") -> Iterator[Dict]:
    ds = load_dataset("lucasmccabe/logiqa", split=split)
    for ex in ds:
        choices = {LETTER[i]: ex[f"option_{i+1}"] for i in range(4)}
        # label is 0..3
        answer = LETTER[int(ex["label"])]
        yield {
            "question": ex["context"] + " " + ex["question"],
            "choices": choices,
            "answer": answer
        }
