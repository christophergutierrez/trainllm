#!/bin/bash
# Full pipeline for APPS 0.5B v2: verify data → train → eval
# Run in a tmux session: tmux new-session -d -s apps05b2 "bash run_apps_0.5b_v2.sh"
set -e

TRAINLLM=~/git_home/trainllm
PYTHON=~/.unsloth/studio/unsloth_studio/bin/python3
HF_HOME=$TRAINLLM/models/hf
CONFIG=$TRAINLLM/config.apps-0.5b-v2.yaml
VERIFIED_DATA=$TRAINLLM/data/apps/train_verified.jsonl
ADAPTER_OUT=$TRAINLLM/lora/apps-0.5b-v2/final
EVAL_OUT=$TRAINLLM/results/apps/target-0.5b-v2

cd $TRAINLLM

# ── Step 1: Build verified training data ─────────────────────────────────────
if [ -f "$VERIFIED_DATA" ]; then
    echo "[$(date '+%H:%M:%S')] $VERIFIED_DATA already exists — skipping verification."
    echo "  Delete it to re-run: rm $VERIFIED_DATA"
else
    echo "[$(date '+%H:%M:%S')] Building verified training data (~30–60 min)..."
    python3 prepare_apps_data.py --verify --output-dir data/apps
    echo "[$(date '+%H:%M:%S')] Verification complete."
fi

# ── Step 2: Kill any lingering GPU processes ─────────────────────────────────
echo "[$(date '+%H:%M:%S')] Checking for GPU processes..."
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
if [ -n "$GPU_PROCS" ]; then
    echo "  Found GPU processes: $GPU_PROCS — sending SIGTERM..."
    echo "$GPU_PROCS" | xargs -r kill -TERM 2>/dev/null || true
    sleep 5
fi

# ── Step 3: Train ─────────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Starting training..."
TRAINLLM_CONFIG=$CONFIG \
HF_HOME=$HF_HOME \
CUDA_LAUNCH_BLOCKING=1 \
    $PYTHON train.py

echo "[$(date '+%H:%M:%S')] Training complete."

# ── Step 4: Eval ─────────────────────────────────────────────────────────────
if [ -d "$ADAPTER_OUT" ]; then
    echo "[$(date '+%H:%M:%S')] Starting eval..."
    HF_HOME=$HF_HOME \
        $PYTHON apps_eval.py \
        --backend hf \
        --model "$ADAPTER_OUT" \
        --test data/apps/eval_all.jsonl \
        --output "$EVAL_OUT" \
        --max-tokens 512 --temp 0.0
    echo "[$(date '+%H:%M:%S')] Eval complete."
    cat "$EVAL_OUT/summary.json"
else
    echo "ERROR: adapter not found at $ADAPTER_OUT — eval skipped."
    exit 1
fi

echo "[$(date '+%H:%M:%S')] Pipeline done. Results at $EVAL_OUT"
