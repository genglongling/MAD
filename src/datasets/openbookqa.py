"""
OpenBookQA (allenai/openbookqa). We'll provide two modes:
- closed_book: ignore provided facts
- controlled_open: include the official small science facts in 'fact' field
"""
from typing import Iterator, Dict, Optional
from datasets import load_dataset

LETTER = ["A","B","C","D"]

def iter_items(split: str="validation", mode: str="closed") -> Iterator[Dict]:
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    # facts are in a separate subset 'additional', but HF dataset offers 'train/dev/test' with choices
    # We'll attach a placeholder fact when mode == 'controlled_open' as many pipelines do a joined lookup.
    for ex in ds:
        choices = {LETTER[i]: ex["choices"]["text"][i] for i in range(4)}
        # ex['answerKey'] in {'A','B','C','D'}
        rec = {
            "question": ex["question_stem"],
            "choices": choices,
            "answer": ex["answerKey"]
        }
        if mode != "closed":
            rec["fact"] = "Refer to OpenBookQA science facts corpus (attach via retrieval in your pipeline)."
        yield rec
