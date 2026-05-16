"""Tests for WSDDecayCallback in train.py.

Uses mock args/state/optimizer to verify LR values at schedule boundaries
without requiring GPU or model loading.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from _callbacks.py — no GPU/torch required
from _callbacks import WSDDecayCallback


def _make_args(lr: float = 1e-3, max_steps: int = 100) -> SimpleNamespace:
    return SimpleNamespace(learning_rate=lr, max_steps=max_steps)


def _make_state(step: int = 0) -> SimpleNamespace:
    return SimpleNamespace(global_step=step)


def _make_optimizer(lr: float = 1e-3):
    return SimpleNamespace(param_groups=[{"lr": lr}])


def _get_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


class TestWSDDecayCallback:
    """stable_ratio=0.7 → stable_end=70, decay from step 71 to 100."""

    def _callback(self, stable_ratio=0.7, min_lr_ratio=0.1, lr=1e-3, max_steps=100):
        cb = WSDDecayCallback(stable_ratio=stable_ratio, min_lr_ratio=min_lr_ratio)
        cb.on_train_begin(_make_args(lr=lr, max_steps=max_steps), _make_state(), SimpleNamespace())
        return cb

    def test_no_change_in_warmup(self):
        cb = self._callback()
        opt = _make_optimizer(1e-3)
        cb.on_step_end(_make_args(), _make_state(step=10), SimpleNamespace(), optimizer=opt)
        assert _get_lr(opt) == pytest.approx(1e-3)

    def test_no_change_at_stable_boundary(self):
        cb = self._callback()
        opt = _make_optimizer(1e-3)
        cb.on_step_end(_make_args(), _make_state(step=70), SimpleNamespace(), optimizer=opt)
        assert _get_lr(opt) == pytest.approx(1e-3)

    def test_decay_starts_after_stable_end(self):
        cb = self._callback()
        opt = _make_optimizer(1e-3)
        cb.on_step_end(_make_args(), _make_state(step=71), SimpleNamespace(), optimizer=opt)
        lr = _get_lr(opt)
        assert lr < 1e-3
        assert lr > 1e-4

    def test_lr_at_midpoint_of_decay(self):
        # stable_end=70, total=100, decay_steps=30
        # step=85 → steps_into_decay=15, fraction=0.5
        # new_lr = 1e-3 * (1 - 0.9 * 0.5) = 5.5e-4
        cb = self._callback()
        opt = _make_optimizer(1e-3)
        cb.on_step_end(_make_args(), _make_state(step=85), SimpleNamespace(), optimizer=opt)
        assert _get_lr(opt) == pytest.approx(5.5e-4, rel=1e-6)

    def test_lr_at_last_step(self):
        # step=100 → steps_into_decay=30=decay_steps, fraction=1.0
        # new_lr = 1e-3 * 0.1 = 1e-4
        cb = self._callback()
        opt = _make_optimizer(1e-3)
        cb.on_step_end(_make_args(), _make_state(step=100), SimpleNamespace(), optimizer=opt)
        assert _get_lr(opt) == pytest.approx(1e-4, rel=1e-6)

    def test_no_change_without_optimizer(self):
        cb = self._callback()
        opt = _make_optimizer(1e-3)
        # on_step_end without optimizer kwarg and no cached optimizer → no-op
        cb.on_step_end(_make_args(), _make_state(step=90), SimpleNamespace())
        assert _get_lr(opt) == pytest.approx(1e-3)  # unchanged

    def test_optimizer_cached_via_on_optimizer_step(self):
        cb = self._callback()
        opt = _make_optimizer(1e-3)
        cb.on_optimizer_step(_make_args(), _make_state(step=85), SimpleNamespace(), optimizer=opt)
        cb.on_step_end(_make_args(), _make_state(step=85), SimpleNamespace())
        assert _get_lr(opt) == pytest.approx(5.5e-4, rel=1e-6)

    def test_monotonic_decay(self):
        cb = self._callback()
        opt = _make_optimizer(1e-3)
        prev_lr = 1e-3
        for step in range(71, 101):
            cb.on_step_end(_make_args(), _make_state(step=step), SimpleNamespace(), optimizer=opt)
            assert _get_lr(opt) <= prev_lr + 1e-10
            prev_lr = _get_lr(opt)

    def test_custom_stable_ratio(self):
        cb = self._callback(stable_ratio=0.5)
        opt = _make_optimizer(1e-3)
        # stable_end = 50, decay_steps = 50
        # no change at step 50
        cb.on_step_end(_make_args(), _make_state(step=50), SimpleNamespace(), optimizer=opt)
        assert _get_lr(opt) == pytest.approx(1e-3)
        # start of decay at step 51
        cb.on_step_end(_make_args(), _make_state(step=51), SimpleNamespace(), optimizer=opt)
        assert _get_lr(opt) < 1e-3

    def test_min_lr_ratio_respected(self):
        cb = self._callback(min_lr_ratio=0.2)
        opt = _make_optimizer(1e-3)
        cb.on_step_end(_make_args(), _make_state(step=100), SimpleNamespace(), optimizer=opt)
        # new_lr = 1e-3 * 0.2 = 2e-4
        assert _get_lr(opt) == pytest.approx(2e-4, rel=1e-6)
