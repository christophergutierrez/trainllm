"""Tests for peft_to_mlx.py — PEFT -> mlx-lm adapter conversion.

Builds tiny synthetic PEFT adapters on disk and checks the things that silently
break inference if wrong: weight key renaming, the (r,in)/(out,r) -> (in,r)/(r,out)
transpose, the scale factor (including the rsLoRA alpha/√r case), and every
refusal path. No GPU and no model download — just torch + safetensors on CPU.
"""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
save_file = safetensors_torch.save_file
load_file = safetensors_torch.load_file

sys.path.insert(0, str(Path(__file__).parent.parent))
import peft_to_mlx
from peft_to_mlx import convert


RANK, IN, OUT = 8, 16, 32


def _layer_weights(layers, modules=("q_proj", "v_proj")):
    """Synthetic PEFT lora_A/lora_B for the given block indices and modules."""
    torch.manual_seed(0)
    w = {}
    for idx in layers:
        for mod in modules:
            base = f"base_model.model.model.layers.{idx}.self_attn.{mod}"
            w[f"{base}.lora_A.weight"] = torch.randn(RANK, IN)   # (r, in)
            w[f"{base}.lora_B.weight"] = torch.randn(OUT, RANK)  # (out, r)
    return w


def _make_adapter(adapter_dir: Path, weights: dict, **config_extra) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"r": RANK, "lora_alpha": 16, "peft_type": "LORA"}
    cfg.update(config_extra)
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg))
    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))


def _convert(tmp_path, weights, **config_extra):
    in_dir = tmp_path / "peft"
    out_dir = tmp_path / "mlx"
    _make_adapter(in_dir, weights, **config_extra)
    convert(in_dir, out_dir)
    out_cfg = json.loads((out_dir / "adapter_config.json").read_text())
    out_w = load_file(str(out_dir / "adapters.safetensors"))
    return out_dir, out_cfg, out_w


class TestBasicConversion:
    def test_keys_renamed_and_transposed(self, tmp_path):
        weights = _layer_weights([0, 1])
        _, cfg, out_w = self._run(tmp_path, weights)

        # PEFT lora_A/lora_B -> mlx lora_a/lora_b under the stripped path.
        assert set(out_w) == {
            f"model.layers.{i}.self_attn.{m}.lora_{ab}"
            for i in (0, 1) for m in ("q_proj", "v_proj") for ab in ("a", "b")
        }
        a = out_w["model.layers.0.self_attn.q_proj.lora_a"]
        b = out_w["model.layers.0.self_attn.q_proj.lora_b"]
        assert tuple(a.shape) == (IN, RANK)   # transposed from (r, in)
        assert tuple(b.shape) == (RANK, OUT)  # transposed from (out, r)

    def test_transpose_preserves_values(self, tmp_path):
        weights = _layer_weights([0])
        orig_a = weights["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
        _, _, out_w = self._run(tmp_path, weights)
        out_a = out_w["model.layers.0.self_attn.q_proj.lora_a"]
        assert torch.equal(out_a, orig_a.t())

    def test_config_fields(self, tmp_path):
        weights = _layer_weights([0, 1])
        _, cfg, _ = self._run(tmp_path, weights)
        assert cfg["fine_tune_type"] == "lora"
        assert cfg["num_layers"] == 2
        assert cfg["lora_parameters"]["rank"] == RANK
        assert cfg["lora_parameters"]["keys"] == ["self_attn.q_proj", "self_attn.v_proj"]

    def test_output_dir_created(self, tmp_path):
        # convert() must create the output directory even when it doesn't exist.
        in_dir = tmp_path / "peft"
        out_dir = tmp_path / "mlx" / "nested"
        _make_adapter(in_dir, _layer_weights([0]))
        assert not out_dir.exists()
        convert(in_dir, out_dir)
        assert out_dir.exists()
        assert (out_dir / "adapters.safetensors").exists()

    def _run(self, tmp_path, weights, **extra):
        return _convert(tmp_path, weights, **extra)


class TestScale:
    def test_plain_lora_scale_is_alpha_over_r(self, tmp_path):
        _, cfg, _ = _convert(tmp_path, _layer_weights([0]))  # alpha=16, r=8
        assert cfg["lora_parameters"]["scale"] == pytest.approx(16 / 8)
        assert cfg["_peft_use_rslora"] is False

    def test_rslora_scale_is_alpha_over_sqrt_r(self, tmp_path):
        _, cfg, _ = _convert(tmp_path, _layer_weights([0]), use_rslora=True)
        assert cfg["lora_parameters"]["scale"] == pytest.approx(16 / (8 ** 0.5))
        assert cfg["_peft_use_rslora"] is True


class TestRefusals:
    def _expect_refusal(self, tmp_path, **config_extra):
        with pytest.raises(SystemExit):
            _convert(tmp_path, _layer_weights([0]), **config_extra)

    def test_rank_pattern(self, tmp_path):
        self._expect_refusal(tmp_path, rank_pattern={"layers.0": 4})

    def test_alpha_pattern(self, tmp_path):
        self._expect_refusal(tmp_path, alpha_pattern={"layers.0": 4})

    def test_use_dora(self, tmp_path):
        self._expect_refusal(tmp_path, use_dora=True)

    def test_bias(self, tmp_path):
        self._expect_refusal(tmp_path, bias="all")

    def test_modules_to_save(self, tmp_path):
        self._expect_refusal(tmp_path, modules_to_save=["embed_tokens"])

    def test_noncontiguous_layers(self, tmp_path):
        # mlx-lm can only target the last N contiguous blocks.
        with pytest.raises(SystemExit):
            _convert(tmp_path, _layer_weights([0, 2]))

    def test_non_tail_range_rejected(self, tmp_path):
        # Layers 0-1 of a 4-layer model: max index is 1, not 3. mlx-lm would
        # apply LoRA to layers 2-3 instead — wrong, so conversion must refuse.
        with pytest.raises(SystemExit):
            _convert(tmp_path, _layer_weights([0, 1]), num_hidden_layers=4)

    def test_tail_range_accepted(self, tmp_path):
        # Layers 2-3 of a 4-layer model: max index is 3 == 4-1. This is the
        # tail, so mlx-lm will correctly target these blocks.
        _, cfg, _ = _convert(tmp_path, _layer_weights([2, 3]), num_hidden_layers=4)
        assert cfg["num_layers"] == 2

    def test_missing_safetensors_raises(self, tmp_path):
        # adapter_config.json present but adapter_model.safetensors absent.
        in_dir = tmp_path / "peft"
        in_dir.mkdir()
        cfg = {"r": RANK, "lora_alpha": 16, "peft_type": "LORA"}
        (in_dir / "adapter_config.json").write_text(json.dumps(cfg))
        with pytest.raises(SystemExit):
            convert(in_dir, tmp_path / "mlx")

    def test_rslora_is_not_refused(self, tmp_path):
        # The whole point of the relaxation: a default-trained adapter converts.
        _, cfg, _ = _convert(tmp_path, _layer_weights([0]), use_rslora=True)
        assert cfg["fine_tune_type"] == "lora"

    def test_missing_config(self, tmp_path):
        in_dir = tmp_path / "peft"
        in_dir.mkdir()
        save_file(_layer_weights([0]), str(in_dir / "adapter_model.safetensors"))
        with pytest.raises(SystemExit):
            convert(in_dir, tmp_path / "mlx")

    def test_no_lora_keys_matched(self, tmp_path):
        with pytest.raises(SystemExit):
            _convert(tmp_path, {"some.random.tensor": torch.randn(4, 4)})


class TestNonBlockModules:
    def test_lm_head_keeps_full_path(self, tmp_path):
        weights = _layer_weights([0])
        weights["base_model.model.lm_head.lora_A.weight"] = torch.randn(RANK, IN)
        weights["base_model.model.lm_head.lora_B.weight"] = torch.randn(OUT, RANK)
        _, cfg, out_w = _convert(tmp_path, weights)
        # Module outside the block stack keeps its full path as the mlx key.
        assert "lm_head.lora_a" in out_w
        assert "lm_head" in cfg["lora_parameters"]["keys"]
        assert cfg["num_layers"] == 1  # only block-stack layers counted
