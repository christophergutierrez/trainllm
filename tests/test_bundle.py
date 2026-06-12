"""Tests for bundle.py — versioning, score gates, and manifest generation."""

import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bundle import bundle, _next_version, _sha256_file, REGISTRY_PATH, MODEL_DIR


@pytest.fixture
def adapter_dir(tmp_path):
    """Create a minimal adapter directory."""
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"fake weights data")
    (d / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "test/model",
        "r": 16, "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "peft_type": "LORA",
    }))
    (d / "tokenizer.json").write_text('{"model": "bpe"}')
    (d / "tokenizer_config.json").write_text('{}')
    (d / "chat_template.jinja").write_text("{{ messages }}")
    return d


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("model: test\n")
    return p


class TestScoreGate:
    def test_rejects_below_min_score(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        eval_info = {"eval_id": "test", "score": 0.5, "band_counts": {}, "num_records": 10}
        result = bundle(adapter_dir, "test-adapter", eval_info, min_score=0.9, cfg_path=cfg_path)
        assert result is None

    def test_passes_above_min_score(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        eval_info = {"eval_id": "test", "score": 0.95, "band_counts": {"EXCELLENT": 10}, "num_records": 10}
        result = bundle(adapter_dir, "test-adapter", eval_info, min_score=0.9, cfg_path=cfg_path)
        assert result is not None
        assert result.exists()

    def test_bundles_without_eval_with_warning(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        result = bundle(adapter_dir, "test-adapter", None, min_score=0.0, cfg_path=cfg_path)
        assert result is not None


class TestTarballContents:
    def test_contains_manifest_and_weights(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        eval_info = {"eval_id": "test", "score": 0.99, "band_counts": {}, "num_records": 5}
        result = bundle(adapter_dir, "my-adapter", eval_info, min_score=0.0, cfg_path=cfg_path)

        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()

        assert "my-adapter-v1/manifest.json" in names
        assert "my-adapter-v1/adapter_model.safetensors" in names
        assert "my-adapter-v1/adapter_config.json" in names
        assert "my-adapter-v1/tokenizer.json" in names
        assert "my-adapter-v1/tokenizer_config.json" in names
        assert "my-adapter-v1/chat_template.jinja" in names

    def test_manifest_content(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        eval_info = {"eval_id": "eval-001", "score": 0.95, "band_counts": {"EXCELLENT": 8}, "num_records": 8}
        result = bundle(adapter_dir, "my-adapter", eval_info, min_score=0.0, cfg_path=cfg_path)

        with tarfile.open(result, "r:gz") as tar:
            f = tar.extractfile("my-adapter-v1/manifest.json")
            manifest = json.loads(f.read())

        assert manifest["schema_version"] == 3
        assert manifest["adapter"] == "my-adapter"
        assert manifest["version"] == 1
        assert manifest["model_name"] == "my-adapter-v1"
        assert manifest["base_model"] == "test/model"
        assert manifest["eval"]["score"] == 0.95
        assert manifest["lora"]["rank"] == 16
        assert "adapter_model.safetensors" in manifest["files"]
        assert "diagnostics" in manifest


class TestReproducibility:
    def test_config_included_in_tarball(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        eval_info = {"eval_id": "t", "score": 0.99, "band_counts": {}, "num_records": 5}
        result = bundle(adapter_dir, "my-adapter", eval_info, min_score=0.0, cfg_path=cfg_path)

        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
            assert "my-adapter-v1/config.yaml" in names
            f = tar.extractfile("my-adapter-v1/config.yaml")
            assert f.read() == b"model: test\n"

    def test_train_data_hash_in_manifest(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        train_data = tmp_path / "training.jsonl"
        train_data.write_text('{"conversations": []}\n')
        eval_info = {"eval_id": "t", "score": 0.99, "band_counts": {}, "num_records": 5}
        result = bundle(adapter_dir, "my-adapter", eval_info, min_score=0.0,
                        cfg_path=cfg_path, train_data_path=train_data)

        with tarfile.open(result, "r:gz") as tar:
            f = tar.extractfile("my-adapter-v1/manifest.json")
            manifest = json.loads(f.read())

        assert manifest["train_data"] is not None
        assert manifest["train_data"]["sha256"]
        assert manifest["train_data"]["size"] > 0
        assert "training.jsonl" in manifest["train_data"]["path"]

    def test_no_train_data_records_none(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        eval_info = {"eval_id": "t", "score": 0.99, "band_counts": {}, "num_records": 5}
        result = bundle(adapter_dir, "my-adapter", eval_info, min_score=0.0,
                        cfg_path=cfg_path, train_data_path=None)

        with tarfile.open(result, "r:gz") as tar:
            f = tar.extractfile("my-adapter-v1/manifest.json")
            manifest = json.loads(f.read())

        assert manifest["train_data"] is None


class TestDiagnostics:
    def test_eval_included_when_evals_dir_provided(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()
        eval_data = {"summary": {"avg_score": 0.99}, "results": [{"score": 0.99}]}
        (evals_dir / "eval-001.json").write_text(json.dumps(eval_data))
        eval_info = {"eval_id": "eval-001", "score": 0.99, "band_counts": {}, "num_records": 1}

        result = bundle(adapter_dir, "my-adapter", eval_info, min_score=0.0,
                        cfg_path=cfg_path, evals_dir=evals_dir)

        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
            assert "my-adapter-v1/diagnostics/eval.json" in names
            f = tar.extractfile("my-adapter-v1/manifest.json")
            manifest = json.loads(f.read())
            assert manifest["diagnostics"]["has_eval"] is True

    def test_convergence_included(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        conv = {"first_loss": 2.8, "final_loss": 0.03, "best_loss": 0.03}
        (adapter_dir / "convergence.json").write_text(json.dumps(conv))
        eval_info = {"eval_id": "t", "score": 0.99, "band_counts": {}, "num_records": 1}

        result = bundle(adapter_dir, "my-adapter", eval_info, min_score=0.0, cfg_path=cfg_path)

        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
            assert "my-adapter-v1/diagnostics/convergence.json" in names
            f = tar.extractfile("my-adapter-v1/manifest.json")
            manifest = json.loads(f.read())
            assert manifest["diagnostics"]["has_convergence"] is True

    def test_loss_history_extracted_from_checkpoint(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        ckpt = adapter_dir / "checkpoint-100"
        ckpt.mkdir()
        trainer_state = {
            "log_history": [
                {"step": 10, "loss": 2.5, "learning_rate": 1e-4},
                {"step": 20, "loss": 1.8, "learning_rate": 1e-4},
                {"step": 30, "loss": 1.2, "learning_rate": 1e-4},
            ]
        }
        (ckpt / "trainer_state.json").write_text(json.dumps(trainer_state))
        eval_info = {"eval_id": "t", "score": 0.99, "band_counts": {}, "num_records": 1}

        result = bundle(adapter_dir, "my-adapter", eval_info, min_score=0.0, cfg_path=cfg_path)

        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
            assert "my-adapter-v1/diagnostics/loss_history.json" in names
            f = tar.extractfile("my-adapter-v1/diagnostics/loss_history.json")
            history = json.loads(f.read())
            assert len(history) == 3
            assert history[0] == [10, 2.5]
            f2 = tar.extractfile("my-adapter-v1/manifest.json")
            manifest = json.loads(f2.read())
            assert manifest["diagnostics"]["has_loss_history"] is True

    def test_no_diagnostics_when_unavailable(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        eval_info = {"eval_id": "t", "score": 0.99, "band_counts": {}, "num_records": 1}

        result = bundle(adapter_dir, "my-adapter", eval_info, min_score=0.0, cfg_path=cfg_path)

        with tarfile.open(result, "r:gz") as tar:
            f = tar.extractfile("my-adapter-v1/manifest.json")
            manifest = json.loads(f.read())
            assert manifest["diagnostics"]["has_eval"] is False
            assert manifest["diagnostics"]["has_convergence"] is False
            assert manifest["diagnostics"]["has_loss_history"] is False


class TestVersioning:
    def test_first_version_is_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        assert _next_version("new-adapter") == 1

    def test_increments_version(self, tmp_path, monkeypatch):
        registry = {"models": [
            {"adapter": "my-adapter", "version": 1},
            {"adapter": "my-adapter", "version": 2},
            {"adapter": "other", "version": 5},
        ]}
        reg_path = tmp_path / "models.json"
        reg_path.write_text(json.dumps(registry))
        monkeypatch.setattr("bundle.REGISTRY_PATH", reg_path)
        assert _next_version("my-adapter") == 3
        assert _next_version("other") == 6
        assert _next_version("brand-new") == 1


class TestRegistry:
    def test_registry_created_on_first_bundle(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        reg_path = tmp_path / "models.json"
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", reg_path)
        eval_info = {"eval_id": "test", "score": 0.99, "band_counts": {}, "num_records": 5}
        bundle(adapter_dir, "test-adapter", eval_info, min_score=0.0, cfg_path=cfg_path)
        assert reg_path.exists()
        data = json.loads(reg_path.read_text())
        assert len(data["models"]) == 1
        assert data["models"][0]["adapter"] == "test-adapter"
        assert data["models"][0]["version"] == 1
        assert data["models"][0]["score"] == 0.99

    def test_multiple_bundles_append(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        reg_path = tmp_path / "models.json"
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", reg_path)
        eval_info = {"eval_id": "t1", "score": 0.91, "band_counts": {}, "num_records": 5}
        bundle(adapter_dir, "adapter-a", eval_info, min_score=0.0, cfg_path=cfg_path)
        eval_info["score"] = 0.95
        bundle(adapter_dir, "adapter-a", eval_info, min_score=0.0, cfg_path=cfg_path)
        data = json.loads(reg_path.read_text())
        assert len(data["models"]) == 2
        assert data["models"][1]["version"] == 2


class TestManifestExtras:
    """chat_template + route_keys surfaced into the manifest for FloCode."""

    def _manifest(self, tarball: Path) -> dict:
        with tarfile.open(tarball, "r:gz") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("manifest.json"))
            return json.loads(tar.extractfile(member).read().decode())

    def test_defaults_when_config_silent(self, adapter_dir, cfg_path, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        # cfg_path fixture declares neither key.
        out = bundle(adapter_dir, "test-adapter", None, min_score=0.0, cfg_path=cfg_path)
        m = self._manifest(out)
        assert m["chat_template"] == "chatml"  # _config default
        assert m["route_keys"] == []

    def test_surfaced_from_config(self, adapter_dir, tmp_path, monkeypatch):
        monkeypatch.setattr("bundle.MODEL_DIR", tmp_path / "models")
        monkeypatch.setattr("bundle.REGISTRY_PATH", tmp_path / "models.json")
        cfg = tmp_path / "config.yaml"
        cfg.write_text("model: test\nchat_template: qwen-2.5\n"
                       "route_keys: [by-id, chained, paginated]\n")
        out = bundle(adapter_dir, "test-adapter", None, min_score=0.0, cfg_path=cfg)
        m = self._manifest(out)
        assert m["chat_template"] == "qwen-2.5"
        assert m["route_keys"] == ["by-id", "chained", "paginated"]
