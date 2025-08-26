
"""
Build a JSONL fact corpus (id,text) from OpenBookQA and QASC via Hugging Face datasets.
Run:
  python scripts/build_fact_index.py --out data/facts/obqa_qasc_facts.jsonl
Requires: datasets, nltk (optional for sentence split).
"""
import argparse, json, pathlib
from datasets import load_dataset

def main(out_path: str):
    out = pathlib.Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        # OpenBookQA facts (the 'additional' set contains science facts)
        try:
            obqa = load_dataset("allenai/openbookqa", "additional", split="train")
            for ex in obqa:
                txt = ex.get("fact") or ex.get("text") or ""
                if txt:
                    f.write(json.dumps({"id": f"obqa-{n}", "text": txt}) + "\n"); n += 1
        except Exception as e:
            print("OBQA additional facts not available:", e)
        # QASC facts
        try:
            qasc = load_dataset("allenai/qasc", split="train")
            for ex in qasc:
                for t in ex.get("fact1", []):
                    f.write(json.dumps({"id": f"qasc-{n}", "text": t}) + "\n"); n += 1
                for t in ex.get("fact2", []):
                    f.write(json.dumps({"id": f"qasc-{n}", "text": t}) + "\n"); n += 1
        except Exception as e:
            print("QASC facts not available:", e)
    print(f"Wrote {n} fact lines to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/facts/obqa_qasc_facts.jsonl")
    args = ap.parse_args()
    main(args.out)
