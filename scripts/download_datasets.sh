#!/usr/bin/env bash
set -e
python -m pip install -q datasets tensorflow-datasets
python scripts/download_hf.py
echo "✅ Datasets downloaded / cached into the Hugging Face cache. Local JSONL snapshots are under data/*/ if export is enabled."
