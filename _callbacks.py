"""Training callbacks used by train.py.

Kept in a separate module so tests can import callbacks without pulling in
torch/unsloth, which require a GPU environment.
"""

import json
import os
from datetime import datetime, timezone

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:  # type: ignore[no-redef]
        pass


EVENTS_FILE = os.environ.get("TRAINLLM_EVENTS", "/tmp/trainllm_events.jsonl")


class EventEmitterCallback(TrainerCallback):
    """Writes structured training events to a JSONL file for the web dashboard."""

    def __init__(self, events_file: str | None = None):
        self._path = events_file or EVENTS_FILE

    def _emit(self, event: dict):
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            pass

    def on_train_begin(self, args, state, control, **kwargs):
        self._emit({"event": "step_start", "step": "train", "max_steps": args.max_steps})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        self._emit({
            "event": "loss",
            "step": state.global_step,
            "value": round(logs["loss"], 5),
            "lr": round(logs.get("learning_rate", 0), 8),
            "epoch": round(logs.get("epoch", 0), 4),
        })

    def on_train_end(self, args, state, control, **kwargs):
        self._emit({"event": "step_end", "step": "train", "duration_sec": None})


class WSDDecayCallback(TrainerCallback):
    """Decay phase of the Warmup-Stable-Decay LR schedule.

    Used with lr_scheduler_type='constant_with_warmup', which handles the
    warmup and stable phases.  This callback linearly decays the LR from
    peak down to min_lr_ratio * peak over the final (1 - stable_ratio)
    fraction of training steps.
    """

    def __init__(self, stable_ratio: float = 0.7, min_lr_ratio: float = 0.1):
        self.stable_ratio  = stable_ratio
        self.min_lr_ratio  = min_lr_ratio
        self._peak_lr: float | None = None
        self._stable_end   = 0
        self._total_steps  = 0
        self._optimizer    = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._peak_lr     = args.learning_rate
        self._total_steps = args.max_steps
        self._stable_end  = int(self.stable_ratio * self._total_steps)

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self._optimizer is None:
            self._optimizer = kwargs.get("optimizer")

    def on_step_end(self, args, state, control, **kwargs):
        if self._peak_lr is None:
            return
        optimizer = kwargs.get("optimizer") or self._optimizer
        if optimizer is None:
            return
        step = state.global_step
        if step <= self._stable_end:
            return
        decay_steps      = max(1, self._total_steps - self._stable_end)
        steps_into_decay = min(step - self._stable_end, decay_steps)
        new_lr = self._peak_lr * (
            1.0 - (1.0 - self.min_lr_ratio) * steps_into_decay / decay_steps
        )
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
