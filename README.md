# Multi‑Agent Debate QA (LangGraph)

This repo runs three debate pairings (GPT‑5 + Claude‑4, GPT‑5 + Gemini‑2.5‑Pro, Claude‑4 + Gemini‑2.5‑Pro) on eight QA datasets and exports a LaTeX table.

## Install
```bash
pip install -e .  # or: pip install -r requirements.txt
```

## Datasets (automatic)
```bash
bash scripts/download_datasets.sh
```
This uses **Hugging Face `datasets`** (and `tfds` when needed) to download/cache:
- ARC‑Challenge (`allenai/ai2_arc`, subset `ARC-Challenge`)
- TruthfulQA‑MC (`EleutherAI/truthful_qa_mc`)
- LogiQA (`lucasmccabe/logiqa`)
- QASC (`allenai/qasc`)
- StrategyQA (`voidful/StrategyQA`)
- OpenBookQA closed‑book & controlled open‑book (`allenai/openbookqa`)
- HellaSwag (`Rowan/hellaswag`)
- GPQA‑Diamond (`fingertap/GPQA-Diamond`, optional)

The script also exports **uniform JSONL snapshots** to `data/*/dev.jsonl` for reproducible runs.

## Run benchmark
```bash
python -m src.runners.run_benchmark   --benchmark configs/benchmark.yaml   --models configs/models.yaml   --datasets configs/datasets.yaml   --prompts configs/prompts.yaml
```

## Export LaTeX table
```bash
python -m src.runners.export_table
# -> results/tables/debate_table.tex
```

## Configs
- `configs/models.yaml`: model vendors & slugs for A/B/Judge
- `configs/datasets.yaml`: points to HF dataset keys (or local paths)
- `configs/prompts.yaml`: debate prompts & contentiousness schedule
- `configs/benchmark.yaml`: pairings, datasets, systems
