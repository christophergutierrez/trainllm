# Troubleshooting

Common issues, their symptoms, and how to fix them.

## Training

### Training hangs (silence timeout)

**Symptom:** `SILENCE TIMEOUT — likely hung — killing` in the cycle log.

**Causes:**
- GPU deadlock (most common on first run after a failed vLLM teardown).
- Model download stalled (first run with a new base model).
- Unsloth Python environment is broken.

**Fix:**
1. Check `dmesg | tail -50` for GPU/CUDA errors.
2. Ensure GPU memory is clean: `python kill_vllm.py` then `nvidia-smi`.
3. If downloading a model for the first time, increase `timeouts.train_silence` in `config.yaml` (default 1800s accommodates most downloads).
4. Verify Unsloth venv: `~/.unsloth/studio/unsloth_studio/bin/python3 -c "from unsloth import FastLanguageModel; print('ok')"`.

### Loss not converging (loss > 1.0 at end of training)

**Symptom:** cycle report says "WARNING: Final loss > 1.0 — model has not converged."

**Causes:**
- Too few `max_steps` for the dataset size.
- Learning rate too low.
- Data too noisy or inconsistent.

**Fix:**
1. Check the step estimate from a canary run: `python cycle.py --canary`.
2. Increase `training.max_steps`. Rule of thumb: 5-10 epochs. With N records and effective batch size B, one epoch = N/B steps.
3. If loss doesn't decrease at all, see the next section.

### Loss decreases but model outputs garbage

**Symptom:** Loss converges to a reasonable value (0.4-0.8) but generated text is incoherent or uses wrong special tokens.

**Cause:** `chat_template` in config.yaml doesn't match the base model family. The model trains on incorrectly tokenized text. Loss drops because it memorizes token patterns, but the output is structurally wrong.

**Fix:** Set `chat_template` to match your model:

| Model family | chat_template |
|-------------|---------------|
| Qwen 2.5 / Qwen 3 | `qwen-2.5` |
| Llama 3.x | `llama-3.1` |
| Gemma | `gemma-it` |
| Mistral | `mistral` |
| Phi-3 | `phi-3` |
| Generic (ChatML) | `chatml` |

### Suspected overfitting (loss < 0.15)

**Symptom:** cycle report says "WARNING: Final loss < 0.15 — may be overfit."

**Fix:**
1. Check holdout scores — if they are high, the warning may be a false alarm (small, clean dataset).
2. If holdout scores are poor: reduce `max_steps`, add more diverse training data, or increase `lora_dropout` from 0 to 0.05.

### Training data validation fails

**Symptom:** `config.yaml is not in ShareGPT format` or similar on startup.

**Fix:** Ensure every line in your training JSONL has this structure:

```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

The system turn is optional. `cycle.py` validates the first 3 records. Common mistakes:
- Using OpenAI messages format (`role`/`content`) instead of ShareGPT (`from`/`value`).
- Empty `conversations` array.
- Malformed JSON (trailing comma, missing quotes).

### OOM during training

**Symptom:** CUDA out-of-memory error during `trainer.train()`.

**Fix (in order of impact):**
1. Reduce `training.batch_size` (e.g., 8 → 2).
2. Reduce `training.max_seq_length` (e.g., 2048 → 1024 if data fits).
3. Enable 4-bit quantization: set `training.load_in_4bit: true`.
4. Ensure vLLM is not running: `python kill_vllm.py --kill`.

### Loss masking produces empty/garbage output

**Symptom:** After enabling `train_on_responses_only: true`, the model generates empty strings or repeats the prompt.

**Cause:** The response boundary markers don't match the actual chat template. `train_on_responses_only` uses string matching to find `<|im_start|>assistant\n` in the tokenized text. If the template differs, no tokens are marked for training.

**Fix:**
1. Verify your chat template produces the expected markers:
   ```python
   tok = get_chat_template(tokenizer, chat_template="qwen-2.5")
   text = tok.apply_chat_template(sample_conversation, tokenize=False)
   print(repr(text))  # look for <|im_start|>assistant\n
   ```
2. If using a non-ChatML model, update the markers in `train.py` (the `instruction_part` and `response_part` arguments to `train_on_responses_only()`).
3. As a quick workaround, set `train_on_responses_only: false` to disable.

### Higher initial loss after enabling training enhancements

**Symptom:** First loss is ~1.0–1.5 instead of the previous ~0.6–0.8.

**Cause:** This is expected when `train_on_responses_only: true`. Previously, loss included easy-to-predict prompt tokens (system messages, user turns) which brought the average down. With response-only masking, loss is computed only on the harder generation tokens.

**Fix:** No fix needed — this is correct behavior. Final converged loss will still reach 0.4–0.8. Compare runs with the same masking setting for valid A/B comparisons.

### Unsloth rejects lora_init value

**Symptom:** `ValueError: Unsloth: init_lora_weights must be either [True, False, "gaussian", "loftq", "corda"].`

**Cause:** The `lora_init` config value is not in Unsloth's allowed list. Notably, `"pissa"` and `"olora"` are supported by PEFT but blocked by Unsloth's wrapper.

**Fix:**
1. Use one of the allowed values: `true`, `false`, `gaussian`, `loftq`, `corda`.
2. To use PiSSA anyway, bypass Unsloth's wrapper (see architecture.md § LoRA initialization).

---

## vLLM / Serving

### vLLM startup timeout

**Symptom:** `FATAL: vLLM startup timeout` after waiting `timeouts.vllm_startup` seconds.

**Causes:**
- `torch.compile` hang on Blackwell/GB10 (missing `--enforce-eager`).
- Not enough GPU memory (training process or leaked vLLM still holding VRAM).
- Model not cached locally (downloading during startup).
- Gated model requires `HF_TOKEN`.

**Fix:**
1. Check that no old processes hold GPU memory: `python kill_vllm.py` then `nvidia-smi`.
2. `--enforce-eager` is set automatically by `cycle.py`. If running vLLM manually, always include it on Blackwell.
3. Increase `timeouts.vllm_startup` if the model is downloading (900s default).
4. For gated models: `export HF_TOKEN=hf_...` before running.

### vLLM serves base model instead of adapter

**Symptom:** Eval delta is ~0 (fine-tuned score equals base score). The cycle log may show: "WARNING: adapter not registered in vLLM."

**Cause:** vLLM started but the LoRA adapter failed to load. Usually because `lora/<adapter>/final/` is missing, corrupt, or has no `.safetensors` files.

**Fix:**
1. Check `lora/<adapter>/final/` exists and contains `adapter_model.safetensors` + `adapter_config.json`.
2. If empty/missing: training didn't save correctly. Re-run training.
3. Verify adapter name in config matches the directory name.

### GPU memory leak after killing vLLM

**Symptom:** `nvidia-smi` shows memory still used after vLLM exits. Next training run OOMs.

**Cause:** On GB10 unified memory, `SIGKILL` leaks CUDA driver memory permanently. Only `SIGTERM` with a clean CUDA teardown releases memory.

**Fix:**
1. Always use `python kill_vllm.py --kill` (sends SIGTERM, waits 30s).
2. Avoid `--force` / SIGKILL unless absolutely necessary.
3. If memory is already leaked, reload NVIDIA drivers:
   ```bash
   sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
   sudo modprobe nvidia nvidia_uvm nvidia_drm nvidia_modeset
   ```
4. If `rmmod` fails (module in use), a reboot is required.

### vLLM process orphans

**Symptom:** After a crash, `ps aux | grep vllm` shows child processes (EngineCore, nccl_heartbeat_monitor, resource_tracker) still running.

**Fix:** `python kill_vllm.py --kill --verify` finds and terminates all vLLM-related processes, including children that survive parent termination.

---

## Evaluation

### All eval scores are 0.0 or ERROR

**Symptom:** Every holdout example scores 0 or shows `[ERROR]` in the report.

**Causes:**
- vLLM is not running or is on a different port.
- Holdout file is in wrong format (ShareGPT instead of messages).
- Model is generating empty responses.

**Fix:**
1. Verify vLLM: `curl http://localhost:8000/v1/models`.
2. Holdout must use `{"messages": [{"role": "user", "content": "..."}]}` format (not `conversations`).
3. Check the eval report `.md` for the actual generated output of failing cases.

### Eval timeout

**Symptom:** `Eval hung after Ns — skipping this model`.

**Cause:** A single vLLM request blocks indefinitely (stuck generation, CUDA hang).

**Fix:**
1. Increase `timeouts.eval_timeout` if the holdout is large (default 3600s).
2. Estimate: ~15s per holdout record for 8B models, ~5s for 1.5B.
3. If one specific record causes hangs, check that record's prompt for unusual length or content.

### Eval scores unexpectedly low due to truncation

**Symptom:** Some records show `length_ratio >> 1.0` and `score < 0.3` despite correct content.

**Cause:** Model generates verbose output that exceeds `max_tokens`, getting truncated. The truncated output scores poorly against the reference.

**Fix:**
1. Check the eval report for `length_ratio` — values > 2.0 suggest verbosity problems.
2. Add concise training examples for the problematic prompt patterns.
3. Increase `max_tokens` in eval.py if the reference answers are legitimately long.

---

## Configuration

### Unknown training keys

**Symptom:** `config.yaml has unknown training keys: {...}. Check for typos.`

**Cause:** A key in the `training:` block is misspelled or not recognized.

**Fix:** Check against the valid keys listed in the error message. Valid training keys:

```
max_seq_length, lora_rank, lora_alpha, lora_dropout,
batch_size, gradient_accumulation_steps, warmup_steps,
max_steps, learning_rate, weight_decay, lr_scheduler,
save_steps, save_total_limit, load_in_4bit,
neftune_noise_alpha, train_on_responses_only,
lora_init, use_rslora
```

Common typos: `learning-rate` (should be `learning_rate`), `max_step` (should be `max_steps`), `nef_tune` (should be `neftune_noise_alpha`).

### Missing required config key

**Symptom:** `config.yaml is missing required key: ...`

**Fix:** Compare your config against `config.example.yaml`. All fields under `model`, `adapter_name`, `paths`, `data`, and `training` are required.

---

## Merging

### "Need at least 2 adapters to merge"

**Fix:** Pass at least two adapter names: `python merge.py --adapters adapter-a,adapter-b`.

### Merge produces degraded output

**Symptom:** Merged model scores lower than individual adapters.

**Cause:** Density too aggressive for low-rank LoRA adapters, or adapters do the same task (better to train a combined adapter).

**Fix:**
1. Increase `density` toward 0.95 (conservative). Default 0.9 is usually fine for LoRA rank 16.
2. If adapters cover the same task on different data, consider retraining on combined data instead of merging.
