# EVINCE: LLM-as-a-judge for Multi-Agent Debate QA via Information Theory 

This repository runs **multi-agent debates** over multiple-choice QA datasets, using **LangGraph** pipelines.  
Each debate consists of **6 rounds** (contentiousness 0.9 → 0.1), with a **judge invoked every round**.  
We log **per-round information-theoretic metrics** and **LLM-based CRIT scores**.
The systems support LangGraph + 6-Round Protocol + LLM-as-a-Judge (CRIT).

---

## 🔹 Features
- **Debate Protocol**: 2 agents (A, B) exchange arguments in 6 rounds, probabilities + rationales per choice.
- **Per-Round Judge**: Independent judge model evaluates outputs after each round, computes CRIT_A / CRIT_B.
- **CRIT Scoring**: LLM-based algorithm using judge prompts to evaluate argument quality and reliability.
- **Metrics**: KL Divergence, JSD, Wasserstein Distance, Mutual Information, Entropy, Information Gain, AvgCRIT.
- **Datasets**: 8 benchmarks (ARC-C, TruthfulQA-MC, LogiQA, QASC, StrategyQA, OpenBookQA Closed, OpenBookQA Controlled Open, HellaSwag, optional GPQA-Diamond).
- **Pairings**:  
  1. Qwen2.5-7B-Instruct vs Qwen2.5-7B-Instruct (self-debate)
  2. Qwen2.5-7B-Instruct vs Llama3.1-8B-Instruct
  3. Llama3.1-8B-Instruct vs Llama3.1-8B-Instruct (self-debate)

---

## 🔹 Installation
```bash
git clone <this-repo>
cd multi-agent-debate-qa
pip install -r requirements.txt
```

### Setting up Local Models
To use the local models (Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct), you need to download them first:

```bash
# Setup both models (~30GB total)
python scripts/setup_local_models.py --all

# Or setup individually
python scripts/setup_local_models.py --qwen   # Qwen2.5-7B-Instruct (~14GB)
python scripts/setup_local_models.py --llama  # Llama3.1-8B-Instruct (~16GB)
```

**Requirements for local models:**
- Sufficient RAM (16GB+ recommended)
- GPU with sufficient VRAM (8GB+ recommended for 7B/8B models)
- Stable internet connection for initial download
- HuggingFace account with access to Llama models (for Llama3.1-8B-Instruct)

**Note for Llama models:** You may need to request access to Llama models on HuggingFace and login with `huggingface-cli login`.

---

## 🔹 Datasets
Download all QA datasets with one command:
```bash
bash scripts/download_datasets.sh
```

This uses HuggingFace Datasets.
Exports uniform JSONL snapshots into data/{dataset}/dev.jsonl.

---

## 🔹 Running Debates
To run all pairings × datasets:

```bash
python3 -m src.runners.run_benchmark \
  --benchmark configs/benchmark.yaml \
  --models configs/models.yaml \
  --datasets configs/datasets.yaml \
  --prompts configs/prompts.yaml
```

To run specific pairings, edit `configs/benchmark.yaml` and uncomment the pairings you want to run.

**Example: Run Qwen vs Llama only:**
```yaml
pairings:
  - qwen_llama
```

Outputs:

results/runs/{pairing}__{dataset}.jsonl → raw per-example with all rounds + judge outputs.

results/metrics/{pairing}__{dataset}.json → aggregated metrics.

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
    models.py        # Model wrappers (OpenAI, Anthropic, Google, Local)
    # LLM-based CRIT scoring (via judge prompts)
  datasets/          # Dataset loaders
  runners/           # Run + export scripts
scripts/
  download_datasets.sh
  setup_local_models.py  # Setup script for local models
  setup_qwen.py      # Setup script for Qwen2.5-7B-Instruct
  setup_llama.py     # Setup script for Llama3.1-8B-Instruct
  build_fact_index.py
results/
  runs/              # Per-example outputs
  metrics/           # Aggregates
  tables/            # LaTeX tables
```