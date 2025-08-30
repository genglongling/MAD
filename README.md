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
- **Datasets**: 7 benchmarks covering arithmetic, medical knowledge, logic, commonsense, and ethical reasoning.
- **Pairings**:  
  1. Qwen2.5-7B-Instruct vs Qwen2.5-7B-Instruct (self-debate), Qwen2.5-7B-Instruct (Judge)
  2. Llama3.1-8B-Instruct vs Llama3.1-8B-Instruct (self-debate), Llama3.1-8B-Instruct (Judge)
  3. Qwen2.5-7B-Instruct vs Llama3.1-8B-Instruct
---

## 🔹 Code Structure

```
MAD/
├── configs/                    # Configuration files
│   ├── benchmark.yaml         # Main benchmark settings
│   ├── datasets.yaml          # Dataset configurations
│   ├── models.yaml            # Model pairings and settings
│   └── prompts.yaml           # Debate and judge prompts
├── data/                      # Local datasets
│   └── arithmetic/
│       └── dev.jsonl          # Custom arithmetic dataset (100 questions)
├── results/                   # Output directory
│   ├── runs/                  # Individual debate results
│   ├── metrics/               # Aggregated metrics
│   └── tables/                # LaTeX tables
├── scripts/                   # Utility scripts
│   ├── setup_local_models.py  # Download local models
│   └── download_datasets.sh   # Download HuggingFace datasets
└── src/                       # Source code
    ├── debate/                # Core debate system
    │   ├── graph.py           # LangGraph debate pipeline
    │   ├── models.py          # Model wrappers (OpenAI, Local, etc.)
    │   ├── prompts.py         # Response parsing and validation
    │   └── metrics.py         # Information-theoretic metrics
    ├── datasets/              # Dataset loaders
    └── runners/               # Execution scripts
        ├── run_benchmark.py   # Main benchmark runner
        └── export_table.py    # Results export
```

---

## 🔹 Installation
```bash
git clone <this-repo>
cd MAD
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

### Supported Datasets (7 total, 1,698 questions)

1. **Arithmetic** (100 questions): Custom arithmetic reasoning dataset
2. **GSM8K** (300 questions): Mathematical word problems
3. **MMLU Professional Medicine** (272 questions): Medical knowledge assessment
4. **MMLU Formal Logic** (126 questions): Logical reasoning problems
5. **HellaSwag** (300 questions): Commonsense reasoning
6. **CommonSenseQA** (300 questions): Commonsense question answering
7. **HH-RLHF** (300 questions): Helpful and Harmless RLHF dataset

### Download Datasets
```bash
# Download HuggingFace datasets
bash scripts/download_datasets.sh

# Custom arithmetic dataset is already included in data/arithmetic/dev.jsonl
```

---

## 🔹 Running Debates

### Quick Start (Single Example)
```bash
# Run with just 1 example per dataset for testing
python -m src.runners.run_benchmark \
  --benchmark configs/benchmark.yaml \
  --models configs/models.yaml \
  --datasets configs/datasets.yaml \
  --prompts configs/prompts.yaml
```

### Full Benchmark Run
```bash
# Run all pairings × datasets (1,698 questions total)
python -m src.runners.run_benchmark \
  --benchmark configs/benchmark.yaml \
  --models configs/models.yaml \
  --datasets configs/datasets.yaml \
  --prompts configs/prompts.yaml
```

### Configuration Options

**Edit `configs/benchmark.yaml` to customize:**
```yaml
# Which pairings to run
pairings:
  - qwen_qwen      # Qwen self-debate
  # - qwen_llama   # Qwen vs Llama
  # - llama_llama  # Llama self-debate

# Which datasets to run
datasets:
  - arithmetic     # Custom arithmetic dataset
  - gsm8k          # Mathematical reasoning
  - mmlu_pro_med   # Medical knowledge
  - mmlu_formal_logic  # Logical reasoning
  - hellaswag      # Commonsense reasoning
  - commonsenseqa  # Commonsense QA
  - hh_rlhf        # Ethical reasoning
```

**Edit `configs/datasets.yaml` to adjust dataset sizes:**
```yaml
# Example: Change GSM8K to use 100 questions instead of 300
gsm8k: {type: hf, name: gsm8k, subset: main, split: test, max_examples: 100}
```

---

## 🔹 Outputs

### Individual Results
```
results/runs/
├── qwen_qwen_arithmetic_arithmetic_1.json
├── qwen_qwen_gsm8k_gsm8k_1.json
└── ...
```

Each file contains:
- Complete debate transcript (6 rounds)
- Agent responses with probabilities and rationales
- Judge evaluations with CRIT scores
- Information-theoretic metrics

### Aggregated Metrics
```
results/metrics/
└── all_results.json  # Summary of all experiments
```

### LaTeX Tables
```bash
# Export accuracy table
python -m src.runners.export_table

# Export per-round metrics
python -m src.runners.export_table --metrics round
```

---

## 🔹 Debate Protocol

### Round Structure
1. **Round 1**: Initial analysis (contentiousness: 0.9)
2. **Round 2**: Confrontational debate (contentiousness: 0.9)
3. **Round 3**: Balanced discussion (contentiousness: 0.7)
4. **Round 4**: Middle ground exploration (contentiousness: 0.5)
5. **Round 5**: Supportive discussion (contentiousness: 0.3)
6. **Round 6**: Final synthesis (contentiousness: 0.1)

### Response Format
Each agent generates:
```json
{
  "output": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
  "reason": {
    "A": "Rationale for choice A",
    "B": "Rationale for choice B",
    "C": "Rationale for choice C",
    "D": "Rationale for choice D"
  }
}
```

### Judge Evaluation
After each round, the judge provides:
```json
{
  "outputA": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
  "outputB": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
  "CRIT_A": 0.75,
  "CRIT_B": 0.85,
  "NOTE_A": "Evaluation of Agent A's arguments",
  "NOTE_B": "Evaluation of Agent B's arguments"
}
```

---

## 🔹 Metrics

### Information-Theoretic Metrics
- **KL Divergence**: Measures disagreement between agents
- **Jensen-Shannon Distance**: Symmetric measure of distribution difference
- **Wasserstein Distance**: Earth mover's distance between distributions
- **Mutual Information**: Information shared between agents
- **Entropy**: Uncertainty in agent responses
- **Information Gain**: Reduction in uncertainty over rounds

### CRIT Scores
- **CRIT_A/CRIT_B**: Judge's evaluation of argument quality (0-1)
- **AvgCRIT**: Average CRIT score across agents

---

## 🔹 Troubleshooting

### Common Issues

**Model Loading Errors:**
```bash
# Reinstall transformers with correct architecture
pip install --upgrade --force-reinstall transformers torch
```

**Memory Issues:**
```bash
# Reduce batch size in configs/benchmark.yaml
num_workers: 1
batch_size: 1
```

**Dataset Loading Errors:**
```bash
# Check HuggingFace access for Llama models
huggingface-cli login
```

### Performance Tips
- Use GPU for local models (8GB+ VRAM recommended)
- Reduce `max_examples` in datasets.yaml for faster testing
- Use `num_workers: 1` for stability with local models

---

## 🔹 Citation

If you use this code, please cite:
```bibtex
@article{evince2024,
  title={EVINCE: LLM-as-a-judge for Multi-Agent Debate QA via Information Theory},
  author={...},
  journal={...},
  year={2024}
}
```