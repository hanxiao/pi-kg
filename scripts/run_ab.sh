#!/usr/bin/env bash
# A/B test: run both prompt variants on the same corpus, same budget, same model.
set -u
DR="${1:-/tmp/mini-corpus}"; BUDGET="${2:-10}"
export LLAMA_URL="${LLAMA_URL:-http://34.80.58.36:8080}"
export EMBED_MODEL=jinaai/jina-embeddings-v5-text-small EMBED_BACKEND=local RERANK_BACKEND=local
export CONTEXT_WINDOW=56320 COMPACTION_RESERVE_TOKENS=20000 COMPACTION_KEEP_RECENT_TOKENS=8000
export THINKING_LEVEL=high
PY=~/searchbox/.venv/bin/python
for V in steps outcome; do
  echo "===== VARIANT $V ====="
  PROMPT_VARIANT=$V $PY run_pi_kg.py --dataroom "$DR" --budget "$BUDGET" --out "out/ab_$V"
done
