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

### 2. Optional Lean verification

Lean verification is optional. If `lean` is unavailable, evaluation still runs
and writes predictions, but `lean_pass` will be `null` for records that were not
compiled.

There are two useful verification levels:

- **Basic standalone verification:** installs the Lean CLI and checks simple
  reconstructed goals, including the bundled fixture. This is enough to confirm
  the harness works.
- **Full project verification:** also downloads/builds the source Lean project
  used by the dataset, so identifiers from Mathlib and the statistical-learning
  formalization are available. This is needed for meaningful compiler checks on
  the real `lean4-stat-learning-theory-novel` examples.

#### Basic: Lean 4 via elan

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

Verify the standalone fixture:

```bash
python3 scripts/lean_verify.py --file eval/lean_harness/Fixture.lean
# expected: {"passed": true, ...}
```

#### Full: source project and Mathlib cache

The real evaluation states reference project-local declarations such as
`coveringNumber`, `GaussianSobolevNormSq`, and `rademacherProductMeasure`.
Those cannot be checked by a bare Lean install. For full verification, clone the
source project outside this repo and let Lake fetch/build its dependencies:

```bash
mkdir -p ~/lean-projects
git clone https://github.com/YuanheZ/lean-stat-learning-theory \
  ~/lean-projects/lean-stat-learning-theory
cd ~/lean-projects/lean-stat-learning-theory
lake exe cache get   # downloads prebuilt Mathlib artifacts when available
lake build           # may take a while if dependencies are not cached
```

Do not commit the clone, `.lake/`, `.elan/`, or downloaded caches. They are local
tooling artifacts only.

### 3. Disk and network

- ~15 GB for `Qwen2.5-Coder-7B-Instruct` (auto-downloaded on first use)
- ~1 GB for `Qwen2.5-Coder-0.5B-Instruct`
- Lean full-project verification additionally uses disk for the external source
  project, Lake packages, and Mathlib build/cache artifacts.
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

This fixture is the basic verification level. If full project verification is
not installed, real dataset examples that mention Mathlib or project-local names
may be skipped or fail with unknown identifiers.

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

### Fused 7B target with retrieved training context

This run tests whether the fine-tuned target still benefits from examples
retrieved from the training split.  If it improves, the model did not fully
internalize some useful facts or proof patterns; if it does not, more of the
value is already baked into the fine-tuned weights.

```bash
python3 scripts/lean_eval.py \
  --model fused-7b-lean \
  --test data/lean_stat/test.jsonl \
  --context-train data/lean_stat/train.jsonl \
  --n-shots 5 \
  --output reports/lean_eval/target-7b-with-context
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

## Step 6: Frontier Baselines (Phase 5.4)

The default frontier path should use an existing ChatGPT/Claude plan manually,
not a token-billed API.  API-based frontier evaluation is an explicit opt-in
path only.

### Plan/UI baseline names

Use these output names when scoring responses collected through a paid plan UI:

- `reports/lean_eval/frontier-no-context`
- `reports/lean_eval/frontier-with-context`

The API runner below is not the default way to create those runs.

### Paid API zero-shot (explicit opt-in)

Requires the Anthropic Python SDK, a valid API key, and `--use-api`.  This may
incur token-based charges that are separate from a ChatGPT or Claude plan.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...

python3 scripts/frontier_api_eval.py --use-api \
  --mode no-context \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/frontier-api-no-context
```

### Paid API few-shot (with training context, explicit opt-in)

Samples 5 examples from the training split as in-context demonstrations
(the same data used to fine-tune the local models).

```bash
python3 scripts/frontier_api_eval.py --use-api \
  --mode with-context \
  --train data/lean_stat/train.jsonl \
  --n-shots 5 \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/frontier-api-with-context
```

Both runs write `predictions.jsonl` with `input_tokens` and `output_tokens`
per record.  The report table will show mean token counts so you can compare
API cost against local inference throughput.

**Cost note:** Each record sends the full state as input tokens.  The
with-context run also prepends 5 training examples, roughly doubling input
token count.  At 1,866 test records this is a non-trivial API spend — run
with `--limit 50` first to spot-check before running the full set.

```bash
# Smoke check before full run (50 records each)
python3 scripts/frontier_api_eval.py --use-api --mode no-context --limit 50 --skip-lean \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/frontier-api-no-context-smoke

python3 scripts/frontier_api_eval.py --use-api --mode with-context --limit 50 --skip-lean \
  --train data/lean_stat/train.jsonl \
  --test data/lean_stat/test.jsonl \
  --output reports/lean_eval/frontier-api-with-context-smoke
```

## Step 7: Generate the report (Phase 5.3)

Run from the bundle root after completing whichever evaluation runs you want
included.  The script auto-detects any `*/summary.json` files under
`reports/lean_eval/` and includes them in the table:

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
For real dataset examples, a bare Lean install is not enough; install the full
source project described in the prerequisites if you need compiler checks against
project-local declarations. The Limitations section of FULL_EVAL.md describes
the remaining reconstruction limits.

**Speculative decoding is slower than target-only**
Expected when draft acceptance rate is low. Check that both `--model` and
`--draft-model` flags are set. If still slower, try `--num-draft-tokens 3`
to reduce draft overhead on shorter tactics.

**HuggingFace download fails**
Ensure you have internet access. For restricted networks set:
`export HF_ENDPOINT=https://hf-mirror.com` or configure a local mirror.
