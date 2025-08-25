"""
Download/cache all datasets via HF datasets & optionally export unified JSONL snapshots under data/*/dev.jsonl.
"""
import os, pathlib, json
from datasets import load_dataset

BASE = pathlib.Path("data")
BASE.mkdir(exist_ok=True, parents=True)

def export_jsonl(ds_iter, path: pathlib.Path, limit=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in ds_iter:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n

# Import our loaders
from src.datasets import arc_c, truthfulqa_mc, logiqa, qasc, strategyqa, openbookqa, hellaswag, gpqa_diamond

def main():
    # Simply iterate through loaders and export a small dev snapshot for quick tests
    total = 0
    total += export_jsonl(arc_c.iter_items("validation"), BASE/"arc_c"/"dev.jsonl", limit=2000)
    total += export_jsonl(truthfulqa_mc.iter_items("validation"), BASE/"truthfulqa_mc"/"dev.jsonl", limit=817)
    total += export_jsonl(logiqa.iter_items("validation"), BASE/"logiqa"/"dev.jsonl", limit=867)
    total += export_jsonl(qasc.iter_items("validation"), BASE/"qasc"/"dev.jsonl", limit=926)
    total += export_jsonl(strategyqa.iter_items("validation"), BASE/"strategyqa"/"dev.jsonl", limit=2290)
    total += export_jsonl(openbookqa.iter_items("validation","closed"), BASE/"openbookqa"/"closed_book"/"dev.jsonl", limit=500)
    total += export_jsonl(openbookqa.iter_items("validation","controlled_open"), BASE/"openbookqa"/"controlled_open_book"/"dev.jsonl", limit=500)
    total += export_jsonl(hellaswag.iter_items("validation"), BASE/"hellaswag"/"dev.jsonl", limit=10042)
    try:
        total += export_jsonl(gpqa_diamond.iter_items("validation"), BASE/"gpqa_diamond"/"dev.jsonl", limit=300)
    except Exception as e:
        print("[optional] GPQA-Diamond not downloaded:", e)
    print(f"Exported ~{total} examples in unified JSONL snapshots.")
if __name__ == "__main__":
    main()
