# EVINCE: LLM-as-a-judge for Multi-Agent Debate QA via Information Theory 

This repository runs **multi-agent debates** over multiple-choice QA datasets, using **LangGraph** pipelines.  
Each debate consists of **6 rounds** (contentiousness 0.9 → 0.1), with a **judge invoked every round**.  
We log **per-round information-theoretic metrics** and **LLM-based CRIT scores**.
The systems support LangGraph + 6-Round Protocol + LLM-as-a-Judge (CRIT).

---

## 🔹 Features
- **Debate Protocol**: 2 agents (A, B) exchange arguments in 6 rounds, probabilities + rationales per choice.
- **Per-Round Judge**: Independent judge model evaluates outputs after each round, computes CRIT_A / CRIT_B.
- **CRIT Scoring**: LLM-based algorithm (with optional rule-base algorithm: BM25 fact index for OBQA/QASC corpora).
- **Metrics**: KL Divergence, JSD, Wasserstein Distance, Mutual Information, Entropy, Information Gain, AvgCRIT.
- **Datasets**: 8 benchmarks (ARC-C, TruthfulQA-MC, LogiQA, QASC, StrategyQA, OpenBookQA Closed, OpenBookQA Controlled Open, HellaSwag, optional GPQA-Diamond).
- **Pairings**:  
  1. GPT-5 vs Claude-4  
  2. GPT-5 vs Gemini-2.5-Pro  
  3. Claude-4 vs Gemini-2.5-Pro  
  4. GPT-5 vs GPT-5 (baseline self-debate)

---

## 🔹 Installation
```bash
git clone <this-repo>
cd multi-agent-debate-qa
pip install -r requirements.txt
```
Dependencies:

langgraph, datasets, rank_bm25, orjson, python-dotenv

plus OpenAI/Anthropic/Google API clients (depending on providers you use)

---
## 🔹 Datasets
Download all QA datasets with one command:
```bash
bash scripts/download_datasets.sh
```

This uses HuggingFace Datasets.
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

Exports uniform JSONL snapshots into data/{dataset}/dev.jsonl.

---

## 🔹 Running Debates
To run all 4 pairings × 8 datasets:

```bash
python -m src.runners.run_benchmark \
  --benchmark configs/benchmark.yaml \
  --models configs/models.yaml \
  --datasets configs/datasets.yaml \
  --prompts configs/prompts.yaml
```

Outputs:

results/runs/{pairing}__{dataset}.jsonl → raw per-example with all rounds outputs.

---

## 🔹 Metrics & Tables
Export LaTeX tables:

```bash
# Accuracy table
python -m src.runners.export_table

# Info-theory metrics (per-round averages)
python -m src.runners.export_table --metrics round
```

Tables are written under results/tables/.

---
## 🔹 Facts Index (Optional, for CRIT)
For OBQA/QASC, you can build a fact index:

```bash
python scripts/build_fact_index.py --out data/facts/obqa_qasc_facts.jsonl
```
Then pass the path in configs/benchmark.yaml under facts_jsonl:.

---
## 🔹 Repo Structure
```bash
configs/
  models.yaml        # Agent pairings
  datasets.yaml      # Dataset configs
  prompts.yaml       # Debate + judge prompts
  benchmark.yaml     # Run all pairings × datasets
src/
  debate/
    graph.py         # Debate pipeline (6 rounds, judge after each round)
    prompts.py       # Parsing + schema validation
    metrics.py       # Info-theoretic metrics
  datasets/          # Dataset loaders
  runners/           # Run + export scripts
scripts/
  download_datasets.sh
  build_fact_index.py
results/
  runs/              # Per-example outputs
  metrics/           # Aggregates
  tables/            # LaTeX tables
```
