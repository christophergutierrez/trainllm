#!/usr/bin/env python3
"""DARE-TIES model merging via mergekit.

Merges multiple LoRA adapters into a single full model by:
  1. Merging each LoRA adapter into the base model (merge_and_unload)
  2. Running mergekit DARE-TIES across the resulting full models

Usage:
    python merge.py                           # use merge config from config.yaml
    python merge.py --adapters a,b,c          # merge specific adapters
    python merge.py --density 0.5             # override density
    python merge.py --dry-run                 # generate mergekit config without running

Requires: mergekit (pip install -e mergekit/)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config

cfg = _config.load()


def find_adapter_dirs(adapter_names: list[str]) -> list[tuple[str, Path]]:
    """Resolve adapter names to their on-disk final/ directories."""
    lora_root = cfg.base_dir / "lora"
    found = []
    for name in adapter_names:
        candidates = [
            lora_root / name / "dpo_final" / "final",
            lora_root / name / "dpo_final",
            lora_root / name / "final",
        ]
        resolved = None
        for p in candidates:
            if p.exists() and (p / "adapter_model.safetensors").exists():
                resolved = p
                break
        if resolved is None:
            print(f"WARNING: adapter '{name}' not found, skipping", file=sys.stderr)
            continue
        found.append((name, resolved))
    return found


def merge_and_unload_adapter(adapter_name: str, adapter_path: Path, output_dir: Path) -> Path:
    """Merge a LoRA adapter into the base model and save as a full model."""
    dest = output_dir / adapter_name
    if dest.exists() and (dest / "config.json").exists():
        print(f"  Reusing existing unloaded model: {dest}")
        return dest

    print(f"  Unloading {adapter_name} from {adapter_path} ...")

    script = f"""
import torch
import peft.import_utils as _piu
_piu.is_torchao_available = lambda: False
import peft.tuners.lora.torchao as _tao
_tao.is_torchao_available = lambda: False
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    {repr(str(cfg.model))},
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
model = PeftModel.from_pretrained(base, {repr(str(adapter_path))}, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
model.save_pretrained({repr(str(dest))}, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained({repr(str(cfg.model))})
tokenizer.save_pretrained({repr(str(dest))})

# Fix extra_special_tokens format for vLLM compatibility
import json as _json
_tc_path = {repr(str(dest))} + "/tokenizer_config.json"
with open(_tc_path) as _f:
    _tc = _json.load(_f)
_est = _tc.get("extra_special_tokens")
if isinstance(_est, list):
    _tc["extra_special_tokens"] = {{tok: tok for tok in _est}}
    with open(_tc_path, "w") as _f:
        _json.dump(_tc, _f, indent=2)

print("  Saved unloaded model to {dest}")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        env = {**os.environ, "HF_HOME": str(cfg.hf_home)}
        result = subprocess.run(
            [str(cfg.unsloth_python), script_path],
            capture_output=True,
            env=env,
            cwd=str(cfg.base_dir),
        )
        if result.returncode != 0:
            print(result.stderr.decode(errors="replace"), file=sys.stderr)
            raise RuntimeError(f"merge_and_unload failed for {adapter_name}")
    finally:
        os.unlink(script_path)

    return dest


def generate_mergekit_config(
    base_model: str,
    model_dirs: list[tuple[str, Path]],
    density: float,
    default_weight: float,
    weights: dict[str, float] | None,
    normalize: bool,
) -> dict:
    """Generate a mergekit YAML config dict for DARE-TIES."""
    models = []
    for name, path in model_dirs:
        w = weights.get(name, default_weight) if weights else default_weight
        models.append({
            "model": str(path),
            "parameters": {
                "weight": w,
                "density": density,
            },
        })

    return {
        "merge_method": "dare_ties",
        "base_model": base_model,
        "models": models,
        "parameters": {
            "normalize": normalize,
            "int8_mask": True,
        },
        "dtype": "bfloat16",
    }


def run_mergekit(config_dict: dict, output_dir: Path, cuda: bool = False) -> None:
    """Run mergekit via its Python API in a subprocess (UNSLOTH_PYTHON)."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required: pip install pyyaml")

    config_path = output_dir.parent / "mergekit_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Mergekit config written to: {config_path}")
    print(f"  Output directory: {output_dir}")

    script = f"""
import torch
import yaml
import pydantic
import mergekit.plan as _plan
for _name in dir(_plan):
    _obj = getattr(_plan, _name)
    if isinstance(_obj, type) and issubclass(_obj, pydantic.BaseModel):
        _obj.model_rebuild()
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions

with open({repr(str(config_path))}) as f:
    config_dict = yaml.safe_load(f)

config = MergeConfiguration.model_validate(config_dict)
run_merge(
    config,
    out_path={repr(str(output_dir))},
    options=MergeOptions(
        lazy_unpickle=True,
        cuda={"True" if cuda else "False"},
        low_cpu_memory=True,
    ),
)
print("Merge complete.")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        env = {**os.environ, "HF_HOME": str(cfg.hf_home)}
        print(f"  Running mergekit via {cfg.unsloth_python}")
        result = subprocess.run(
            [str(cfg.unsloth_python), script_path],
            capture_output=True,
            env=env,
        )
        if result.returncode != 0:
            print(result.stderr.decode(errors="replace"), file=sys.stderr)
            raise RuntimeError(f"mergekit failed (exit {result.returncode})")
    finally:
        os.unlink(script_path)

    print(f"\n  Merged model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapters", help="Comma-separated adapter names to merge (default: from config)")
    parser.add_argument("--density", type=float, help="Override DARE density parameter")
    parser.add_argument("--weight", type=float, help="Override default per-model weight")
    parser.add_argument("--output", help="Override output directory for merged model")
    parser.add_argument("--cuda", action="store_true", help="Use GPU acceleration for mergekit")
    parser.add_argument("--dry-run", action="store_true", help="Print mergekit config without running")
    parser.add_argument("--clean", action="store_true", help="Remove intermediate unloaded models after merge")
    args = parser.parse_args()

    if cfg.merge is None:
        if not args.adapters:
            raise SystemExit(
                "No merge config in config.yaml and no --adapters specified.\n"
                "Add a 'merge:' section to config.yaml or pass --adapters a,b,c"
            )
        adapter_names = [a.strip() for a in args.adapters.split(",")]
        density = args.density or 0.9
        default_weight = args.weight or 1.0
        normalize = True
        output_dir = Path(args.output) if args.output else cfg.base_dir / "merged" / "default"
        adapter_weights = None
    else:
        if args.adapters:
            adapter_names = [a.strip() for a in args.adapters.split(",")]
        elif cfg.merge.adapters == "all":
            lora_root = cfg.base_dir / "lora"
            adapter_names = [
                d.name for d in sorted(lora_root.iterdir())
                if d.is_dir() and (d / "final" / "adapter_model.safetensors").exists()
            ]
        else:
            adapter_names = [a["name"] for a in cfg.merge.adapters]

        density = args.density or cfg.merge.density
        default_weight = args.weight or cfg.merge.weight
        normalize = cfg.merge.normalize
        output_dir = Path(args.output) if args.output else Path(cfg.merge.output_dir)
        adapter_weights = {
            a["name"]: a.get("weight", default_weight)
            for a in (cfg.merge.adapters if cfg.merge.adapters != "all" else [])
        }

    print("=" * 60)
    print("  DARE-TIES Model Merge")
    print("=" * 60)
    print(f"  Base model:  {cfg.model}")
    print(f"  Adapters:    {adapter_names}")
    print(f"  Density:     {density}")
    print(f"  Weight:      {default_weight}")
    print(f"  Normalize:   {normalize}")
    print(f"  Output:      {output_dir}")
    print()

    # Step 1: Find adapter directories
    adapters = find_adapter_dirs(adapter_names)
    if len(adapters) < 2:
        raise SystemExit(f"Need at least 2 adapters to merge, found {len(adapters)}")

    # Step 2: merge_and_unload each adapter
    print("Step 1: Merging LoRA adapters into base model (merge_and_unload)")
    print("-" * 60)
    unload_dir = cfg.base_dir / "merged" / "_unloaded"
    unload_dir.mkdir(parents=True, exist_ok=True)

    unloaded_models = []
    for name, adapter_path in adapters:
        model_path = merge_and_unload_adapter(name, adapter_path, unload_dir)
        unloaded_models.append((name, model_path))

    # Step 3: Generate mergekit config
    print("\nStep 2: Running DARE-TIES merge via mergekit")
    print("-" * 60)
    config_dict = generate_mergekit_config(
        base_model=cfg.model,
        model_dirs=unloaded_models,
        density=density,
        default_weight=default_weight,
        weights=adapter_weights,
        normalize=normalize,
    )

    if args.dry_run:
        import yaml
        print("\nMergekit config (dry run):")
        print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
        return

    # Step 4: Run mergekit
    output_dir.mkdir(parents=True, exist_ok=True)
    run_mergekit(config_dict, output_dir, cuda=args.cuda)

    # Step 5: Cleanup intermediate models if requested
    if args.clean:
        print(f"\nCleaning up intermediate models: {unload_dir}")
        shutil.rmtree(unload_dir)

    print("\n" + "=" * 60)
    print("  Merge complete!")
    print(f"  Model: {output_dir}")
    print("  Serve: vllm serve " + str(output_dir) + " --dtype bfloat16")
    print("=" * 60)


if __name__ == "__main__":
    main()
