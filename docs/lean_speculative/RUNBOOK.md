# Lean Speculative Decoding — Runbook

Step-by-step instructions for running the full evaluation on a Mac with Apple
Silicon. All commands run from the **bundle root** (`lean-speculative-bundle/`).

Scripts live in `scripts/` inside the bundle. Run them as
`python3 scripts/<name>.py`, not `python3 <name>.py`.

## Prerequisites

### 1. Python with mlx-lm

```bash
pip install "mlx-lm>=0.21.0"
```

Verify speculative decoding is available:

```bash
python3 -m mlx_lm.generate --help | grep draft
# expected: --draft-model  and  --num-draft-tokens
```

### 2. Lean 4 via elan

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source ~/.elan/env
lean --version
```

Add elan to your shell profile so `lean` is available in new terminal sessions:

```bash
echo 'source ~/.elan/env' >> ~/.zshrc   # zsh (default on Mac)
# or
echo 'source ~/.elan/env' >> ~/.bashrc  # bash
```

### 3. Disk and network

- ~15 GB for `Qwen2.5-Coder-7B-Instruct` (auto-downloaded on first use)
- ~1 GB for `Qwen2.5-Coder-0.5B-Instruct`
- HuggingFace account is not required for public models, but a VPN or firewall
  may block downloads. Set `HF_ENDPOINT` or `HUGGINGFACE_HUB_URL` if needed.

## Step 1: Copy MLX adapters from the training machine

The GB10 training machine produced PEFT adapters that were converted to MLX
format. Copy them to the Mac:

```bash
rsync -av gb10:~/git_home/trainllm/adapters/0.5b-lean-mlx/ \
  ~/trainllm-repo/adapters/0.5b-lean-mlx/

rsync -av gb10:~/git_home/trainllm/adapters/7b-lean-mlx/ \
  ~/trainllm-repo/adapters/7b-lean-mlx/
```

Verify both contain `adapters.safetensors` and `adapter_config.json`.

## Step 2: Fuse adapters into standalone MLX models (Phase 3.3–3.4)

Run from inside the bundle root, placing fused models into the placeholders:

```bash
python3 -m mlx_lm.fuse \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --adapter-path ~/trainllm-repo/adapters/0.5b-lean-mlx \
  --save-path fused-0.5b-lean

python3 -m mlx_lm.fuse \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path ~/trainllm-repo/adapters/7b-lean-mlx \
  --save-path fused-7b-lean
```

Smoke-check both fused models:

```bash
python3 -m mlx_lm.generate \
  --model fused-0.5b-lean \
  --prompt "Given the Lean 4 state:
n : Nat
⊢ n + 0 = n
Provide the next tactical step." \
  --max-tokens 32 --temp 0

python3 -m mlx_lm.generate \
  --model fused-7b-lean \
  --prompt "Given the Lean 4 state:
n : Nat
⊢ n + 0 = n
Provide the next tactical step." \
  --max-tokens 32 --temp 0
```

Both should return something like `simp` or `omega`, not an empty string.

## Step 3: Verify Lean harness (Phase 4)

Safety-check only (no Lean needed):

```bash
python3 scripts/lean_verify.py --check "simp"    # → SAFE
python3 scripts/lean_verify.py --check "sorry"   # → FORBIDDEN
```

Compile the fixture to confirm `lean` is in PATH:

```bash
python3 scripts/lean_verify.py --file eval/lean_harness/Fixture.lean
# expected: {"passed": true, ...}
```

Run the 5-record fixture through the full pipeline:

```bash
python3 scripts/lean_eval.py \
  --model fused-7b-lean \
  --test data/sample/fixture_5.jsonl \
  --output reports/lean_eval/fixture
```

## Step 4: Baseline evaluation runs (Phase 5.1)

### Base 7B (untuned)

```bash
python3 scripts/lean_eval.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/base-7b
```

### Fused 7B target

```bash
python3 scripts/lean_eval.py \
  --model fused-7b-lean \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/target-7b
```

### Fused 0.5B draft alone

```bash
python3 scripts/lean_eval.py \
  --model fused-0.5b-lean \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/draft-0.5b
```

## Step 5: Speculative run (Phase 5.2)

```bash
python3 scripts/lean_eval.py \
  --model fused-7b-lean \
  --draft-model fused-0.5b-lean \
  --num-draft-tokens 5 \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/speculative
```

## Step 6: Generate the report (Phase 5.3)

Run from the bundle root — the script auto-detects paths relative to its
location:

```bash
python3 scripts/make_report.py
# writes reports/REPORT.md
cat reports/REPORT.md
```

## Quick smoke run (skip Lean, 20 records)

Use `--limit 20 --skip-lean` for a fast end-to-end pipeline check:

```bash
python3 scripts/lean_eval.py --model fused-7b-lean \
  --test data/sample/test_20.jsonl --limit 20 --skip-lean \
  --output reports/lean_eval/smoke

python3 scripts/lean_eval.py --model fused-7b-lean \
  --draft-model fused-0.5b-lean --num-draft-tokens 5 \
  --test data/sample/test_20.jsonl --limit 20 --skip-lean \
  --output reports/lean_eval/smoke-speculative
```

## Troubleshooting

**`python3: can't open file 'lean_eval.py'`**
You are running from the bundle root but forgot the `scripts/` prefix.
Use `python3 scripts/lean_eval.py`, not `python3 lean_eval.py`.

**`mlx_lm.generate` crashes with `TypeError: unexpected keyword argument 'verbose'`**
Your mlx-lm is older than 0.21.0. Upgrade: `pip install "mlx-lm>=0.21.0"`.

**`mlx_lm.generate` crashes with "weight not found"**
The PEFT → MLX conversion may have produced misaligned keys. Rerun
`peft_to_mlx.py` on the training machine and re-sync the adapter directories.

**Lean verification always returns `lean_ok: null`**
Either `lean` is not in PATH (open a new terminal and run `source ~/.elan/env`
or add it to your shell profile), or the goal state cannot be reconstructed as
a standalone `example` (complex Mathlib states with `inst✝` or metavariables).
The Limitations section of FULL_EVAL.md describes this in detail.

**Speculative decoding is slower than target-only**
Expected when draft acceptance rate is low. Check that both `--model` and
`--draft-model` flags are set. If still slower, try `--num-draft-tokens 3`
to reduce draft overhead on shorter tactics.

**HuggingFace download fails**
Ensure you have internet access. For restricted networks set:
`export HF_ENDPOINT=https://hf-mirror.com` or configure a local mirror.
