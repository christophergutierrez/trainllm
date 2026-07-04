# Magicoder / HumanEval+ Demo

Goal: demonstrate that supervised fine-tuning improves code-generation quality,
then measure whether a smaller draft model speeds up target-model generation.

## Data

- Training source: `ise-uiuc/Magicoder-OSS-Instruct-75K`
- Primary benchmark: `evalplus/humanevalplus`
- Reference benchmark: `openai/openai_humaneval`

`prepare_magicoder_data.py` filters Magicoder to Python solutions, strips
markdown fences/explanations from assistant targets, writes role/content chat
records into `data/magicoder/train.jsonl` and `data/magicoder/holdout.jsonl`,
then writes both HumanEval and HumanEval+ test files. HumanEval+ is the primary
benchmark because standard HumanEval is too easy for
`Qwen/Qwen2.5-Coder-7B-Instruct` in this prompt format.

## Pipeline

Run:

```bash
tmux new-session -d -s magicoder "bash run_magicoder_pipeline.sh 2>&1 | tee /tmp/magicoder.log"
```

The pipeline order is:

1. Prepare Magicoder and HumanEval+ data.
2. Evaluate base 0.5B and base 7B on HumanEval+.
3. Train and evaluate the 0.5B draft adapter.
4. Stop if the 0.5B fine-tune does not beat the 0.5B baseline.
5. Train and evaluate the 7B target adapter only after that canary passes.

This keeps failed data or prompt choices from burning a full 7B training run.

The cleaned run uses `magicoder-0.5b-py` and `magicoder-7b-py` adapter names,
plus `*-plus-clean` result directories, so stale adapters and summaries from
the earlier mixed-language run are not reused.
