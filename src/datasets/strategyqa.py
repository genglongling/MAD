"""
StrategyQA (voidful/StrategyQA). Binary yes/no -> map to A/B.
"""
from typing import Iterator, Dict
from datasets import load_dataset

def iter_items(split: str="validation") -> Iterator[dict]:
    ds = load_dataset("voidful/StrategyQA", split=split)
    for ex in ds:
        # 'question','answer' (bool)
        choices = {"A":"Yes","B":"No","C":"", "D":""}
        answer = "A" if bool(ex["answer"]) else "B"
        yield {
            "question": ex["question"],
            "choices": choices,
            "answer": answer
        }
