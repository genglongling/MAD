"""
TruthfulQA multiple-choice (EleutherAI/truthful_qa_mc)
We use 'mc1' split, with four options; answer is index -> map to A/B/C/D.
"""
from typing import Iterator, Dict
from datasets import load_dataset

LETTER = ["A","B","C","D"]

def iter_items(split: str="validation") -> Iterator[Dict]:
    ds = load_dataset("EleutherAI/truthful_qa_mc", split=split)  # has 'question','mc1_targets','mc1_labels'
    for ex in ds:
        options = ex["mc1_targets"]["choices"]
        # Some items may have >4 choices; take first 4 consistently for MC setting
        opts = options[:4]
        choices = {LETTER[i]: opts[i] for i in range(len(opts))}
        # mc1_labels is a list of indices for correct choices; take first if multiple
        if ex["mc1_labels"]:
            idx = int(ex["mc1_labels"][0])
        else:
            idx = 0
        idx = max(0, min(idx, len(opts)-1))
        answer = LETTER[idx]
        yield {
            "question": ex["question"],
            "choices": choices,
            "answer": answer
        }
