# Data Preparation

End-to-end guide for going from raw endpoint data to training-ready JSONL files.

## Source data format

Each API endpoint lives in its own subdirectory with a `training.jsonl` file. Each record is a question + the correct API call:

```json
{
  "question": "List the first 10 measurements",
  "api_call": {
    "endpoint": "GET /external/v1/measurements",
    "params": {"pageSize": 10}
  },
  "thinking": "Entity: measurement\nScope: list\nUse: GET /external/v1/measurements\nParams: {\"pageSize\": 10}"
}
```

- `question` and `api_call` are required.
- `thinking` is optional. If present, it becomes a `<think>` trace in the training output.
- `api_call` can be a single call (`endpoint` + `params`) or a two-step chain (`steps` array).

### Two-step chain format

```json
{
  "question": "Get measurement 42",
  "api_call": {
    "steps": [
      {"endpoint": "GET /external/v1/measurements", "params": {}},
      {"endpoint": "GET /external/v1/measurements/{id}", "params": {"id": "{{steps.0.id}}"}}
    ]
  }
}
```

## Directory layout

```
source_data/acme/
├── measurements/
│   └── training.jsonl
├── audiences/
│   └── training.jsonl
├── networks/
│   └── training.jsonl
└── ...
```

`prepare_data.py` discovers all `*/training.jsonl` files under the input directory.

## Basic usage

```bash
python prepare_data.py \
    --input-dir ~/source_data/acme \
    --train-out  ~/trainLLM/data/training.jsonl \
    --holdout-out ~/trainLLM/data/holdout.jsonl \
    --org-name acme
```

This produces two files:
- **Training JSONL** (ShareGPT format) — fed to `train.py` via Unsloth.
- **Holdout JSONL** (OpenAI messages format) — fed to `eval.py` via vLLM.

### Dry run

```bash
python prepare_data.py --input-dir ~/source_data/acme --dry-run
```

Prints per-endpoint record counts and split sizes without writing files.

## Output formats

### Training (ShareGPT)

```json
{"conversations": [
    {"from": "system", "value": "You are an acme API assistant. ..."},
    {"from": "human",  "value": "List the first 10 measurements"},
    {"from": "gpt",    "value": "<think>\nEntity: measurement\n...\n</think>\n```json\n{...}\n```"}
]}
```

### Holdout (OpenAI messages)

```json
{
  "id": "measurements-0042",
  "label": "List the first 10 measurements",
  "messages": [
    {"role": "system",    "content": "You are an acme API assistant. ..."},
    {"role": "user",      "content": "List the first 10 measurements"},
    {"role": "assistant", "content": "<think>\n...\n</think>\n```json\n{...}\n```"}
  ],
  "conventions_tested": ["measurements", "page-size-only"]
}
```

Convention tags are auto-generated from endpoint name and parameter structure (pagination, filtered, no-params, path-param).

## Stratified splits

`prepare_data.py` splits per-endpoint, not globally. Every endpoint gets at least 1 holdout record regardless of size. Default holdout fraction is 10% (`--holdout-frac 0.10`). The `--seed` flag (default 42) ensures reproducible splits.

## System prompt styles

The `--prompt-style` flag controls the system prompt injected into every record.

| Style | Tokens | Best for | Example |
|-------|--------|----------|---------|
| `conversational` (default) | ~60 | Models <=8B | "You are an acme API assistant. Given a natural language request, respond with the correct API call..." |
| `structural` | ~25 | Models 27B+ | "acme API. Plan reasoning in \<think\> tags. Output: JSON code block." |
| `qoc` | ~45 | QOC trace format | "acme API assistant. Reason in \<think\> tags, then output a JSON code block." |

Smaller models benefit from the longer conversational prompt because it provides more grounding context. Larger models can infer the task from shorter structural cues.

Override precedence: CLI `--prompt-style` > trace style match (qoc forces qoc prompt) > `TRAINLLM_PROMPT_STYLE` env var > `conversational`.

## Thinking traces

Thinking traces teach the model to reason before generating JSON. They appear as `<think>...</think>` blocks before the JSON code block in the assistant response.

### Linear format (default)

Structured as Entity/Scope/Use/NOT key-value lines:

```
Entity: measurement
Scope: list -- no specific ID mentioned
Use:    GET /external/v1/measurements
NOT:    GET /external/v1/measurements/{id} (single-item endpoint)
Params: {"pageSize": 10}
```

Key fields: `Entity`, `Scope`, `Use`, `NOT`, `Params`, `Domain`, `Endpoint`, `Filters`, `Goal`, `Step 0`/`Step 1` (for chains), `Requested count`, `Two-step chain`, `Possession note`.

### QOC format (Question/Option/Criteria)

Forces explicit option rejection via a decision matrix:

```
Question: Retrieve a single resource or list all?
Option A: GET /external/v1/measurements/{id}  (single item)
Option B: GET /external/v1/measurements       (list)
Criteria: No explicit ID in the prompt. Option B wins.
Params:   {"pageSize": 10}
```

QOC converts linear traces automatically using detection rules:
- Chain traces (has `Goal`/`Step 0`) -> chain question pattern
- Synonym traces (has `Domain`) -> endpoint disambiguation pattern
- By-ID traces (has `NOT`) -> single vs. list question
- Simple traces -> basic endpoint selection

### Choosing a trace format

```bash
# Linear (default):
python prepare_data.py --input-dir ... --trace-style linear

# QOC:
python prepare_data.py --input-dir ... --trace-style qoc
```

**Linear** is generally better for <=8B models. It is more compact and easier for smaller models to learn.

**QOC** forces explicit rejection of wrong options and was designed for synonym-resistance problems (e.g., "campaign" mapping to `/measurements` instead of a non-existent `/campaigns`). It regressed at 8B in testing but may help at 27B+ where models can reliably execute the decision-matrix logic.

Records without a `thinking` field in the source data produce responses with no `<think>` block.

## Per-endpoint automation

`endpoint_runner.py` automates the full prepare-train-eval loop across multiple endpoints:

```bash
# Prepare all endpoints, then train and eval each:
python endpoint_runner.py --dataset acme --source-root ~/source_data/acme

# Prepare only (generate configs and data, no training):
python endpoint_runner.py --dataset acme --source-root ~/source_data/acme --prepare-only

# Specific endpoints only:
python endpoint_runner.py --dataset acme --endpoint measurements --endpoint audiences
```

For each endpoint, it:
1. Copies source data to durable storage under `data/<dataset>/`.
2. Runs `stratified_split()` and writes per-endpoint train/holdout files to `data/<dataset>_prepared/`.
3. Generates a per-endpoint `config.yaml` with auto-sized `max_steps` based on record count.
4. Calls `cycle.py` with the generated config.

## Holdout preservation

Hand-curated holdout records (prefixed `canonical-`, `mcp-`, `mcp_`) from prior runs are preserved across re-preparations. Their system prompts are updated to the current style, and thinking traces are converted if the trace style changed.

## Handoff directories

Versioned snapshots of prepared data are stored in `handoff/<org>_thinking_<date>_v<N>/`:

```
handoff/videoamp_thinking_20260515_v7/
├── config.qwen3-thinking.yaml     # config snapshot for reproducibility
├── training.jsonl                  # global training set
├── holdout.jsonl                   # global holdout set
└── videoamp/                       # per-endpoint source data
    ├── measurements/training.jsonl
    ├── audiences/training.jsonl
    └── ...
```

These serve as checkpoints: if a training run regresses, you can roll back to a previous handoff's data.

## Validation

`cycle.py` validates training data before spending GPU time. It checks the first 3 records for:
- Valid JSON on each line
- `conversations` key present (not `messages`)
- `conversations` is a non-empty list
- First conversation turn has `from` and `value` keys

It also warns if the dataset is small (< 100 records: "will likely underfit"; 100-500: "consider adding more").
