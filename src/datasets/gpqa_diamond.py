"""
GPQA-Diamond (fingertap/GPQA-Diamond). 4-way MC in the diamond split.
"""
from typing import Iterator, Dict
from datasets import load_dataset

LETTER = ["A","B","C","D"]

def iter_items(split: str="validation") -> Iterator[Dict]:
    ds = load_dataset("fingertap/GPQA-Diamond", split=split)
    for ex in ds:
        # Fields: 'question','options' (list of 4), 'answer' (string exact option) OR index\n
        opts = ex["options"]
        choices = {LETTER[i]: opts[i] for i in range(4)}
        # Ground truth can be index or text depending on version; try both
        ans = ex.get("answer")
        if isinstance(ans, int):
            answer = LETTER[ans]
        elif isinstance(ans, str):
            # map text back to letter
            try:
                idx = opts.index(ans)
                answer = LETTER[idx]
            except ValueError:
                answer = "A"
        else:
            answer = "A"
        yield {
            "question": ex["question"],
            "choices": choices,
            "answer": answer
        }
