#!/usr/bin/env python3
"""
Convert a PEFT/trainllm LoRA adapter to mlx-lm's native adapter format.

mlx-lm does NOT load PEFT adapters (verified against mlx_lm/tuner/utils.py
load_adapters): it expects its own file name, config schema, weight key naming,
and matrix orientation. Worse, it loads weights with strict=False, so a naive
copy of a PEFT adapter "loads" silently and runs the BASE model with a dead
adapter. This script does the real conversion:

  PEFT                                          mlx-lm
  ----------------------------------------      --------------------------------
  adapter_model.safetensors                  -> adapters.safetensors
  base_model.model.<path>.lora_A.weight      -> <path>.lora_a   (TRANSPOSED)
  base_model.model.<path>.lora_B.weight      -> <path>.lora_b   (TRANSPOSED)
  adapter_config.json {r, lora_alpha, ...}   -> adapter_config.json
                                                {fine_tune_type, num_layers,
                                                 lora_parameters{rank, scale,
                                                 dropout, keys}}
  scaling alpha/r (or alpha/√r if rslora)    -> single lora_parameters.scale

Shapes (why the transpose):
  PEFT lora_A: (r, in)   -> mlx lora_a: (in, r)
  PEFT lora_B: (out, r)  -> mlx lora_b: (r, out)
Both forwards then compute x @ A @ B * scale with identical math.

Usage:
  python peft_to_mlx.py --in lora/example-local/final --out lora/example-local/mlx

Verify (MANDATORY before shipping — catches any residual orientation issue):
  vLLM side : temp-0 generation with the adapter on one holdout prompt
  MLX side  : mlx_lm.generate --model <base> --adapter-path <out> \
                --prompt "<same prompt>" --temp 0
  The outputs should match token-for-token on a learned pattern. If MLX output
  equals the BASE model's output instead, keys didn't match (check stderr
  warnings). If it's garbage, orientation/scale is wrong — file an issue.

Refuses to convert (rare PEFT features mlx-lm cannot represent or verify):
  rank_pattern / alpha_pattern (per-layer ranks), use_dora,
  non-empty modules_to_save, bias != "none", unknown total layer count unless
  --allow-unknown-layer-count is passed.

use_rslora IS supported: with a uniform rank (rank_pattern is refused above),
rsLoRA's only change is the scale factor alpha/√r instead of alpha/r — a single
constant that mlx-lm's lora_parameters.scale represents exactly. trainllm's
default config trains with use_rslora=true, so this is the common case.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PEFT_KEY = re.compile(r"^base_model\.model\.(?P<path>.+)\.lora_(?P<ab>[AB])\.weight$")
LAYER_IDX = re.compile(r"^model\.layers\.(?P<idx>\d+)\.(?P<rel>.+)$")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_peft_config(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        fail(f"{cfg_path} not found — is this a PEFT adapter directory?")
    cfg = json.loads(cfg_path.read_text())

    # Features mlx-lm's single-scale, uniform-rank LoRA cannot represent.
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        fail("adapter uses rank_pattern/alpha_pattern (per-layer ranks); "
             "mlx-lm supports one rank+scale. Retrain without patterns or merge instead.")
    if cfg.get("use_dora"):
        fail("DoRA adapters need mlx-lm's DoRA path; this script handles plain LoRA only.")
    if cfg.get("bias", "none") != "none":
        fail(f"bias={cfg['bias']!r} not supported; retrain with bias='none' or merge.")
    if cfg.get("modules_to_save"):
        fail("modules_to_save present (full-weight modules); merge instead.")
    return cfg


def convert(in_dir: Path, out_dir: Path, allow_unknown_layer_count: bool = False) -> None:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        fail("safetensors.torch not found. Install train/conversion dependencies first.")

    peft_cfg = load_peft_config(in_dir)
    weights_path = in_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        fail(f"{weights_path} not found.")
    peft_weights = load_file(str(weights_path))

    mlx_weights: dict[str, Any] = {}
    rel_keys: set[str] = set()
    layer_indices: set[int] = set()
    skipped: list[str] = []

    for key, tensor in peft_weights.items():
        m = PEFT_KEY.match(key)
        if not m:
            skipped.append(key)
            continue
        path, ab = m.group("path"), m.group("ab")
        # PEFT (r,in)/(out,r) -> mlx (in,r)/(r,out)
        mlx_weights[f"{path}.lora_{ab.lower()}"] = tensor.t().contiguous()

        lm = LAYER_IDX.match(path)
        if lm:
            layer_indices.add(int(lm.group("idx")))
            rel_keys.add(lm.group("rel"))
        else:
            # Module outside the block stack (e.g. lm_head). mlx-lm matches
            # these via model.named_modules(); keep the full path as the key.
            rel_keys.add(path)

    if not mlx_weights:
        fail("no lora_A/lora_B weights matched — unexpected key format. "
             f"First keys seen: {list(peft_weights)[:3]}")
    if skipped:
        print(f"WARNING: {len(skipped)} non-LoRA keys skipped: {skipped[:5]}",
              file=sys.stderr)

    # mlx-lm applies LoRA to the LAST num_layers blocks. If PEFT trained a
    # non-tail subset, the converted config would target the wrong blocks.
    n_layers = len(layer_indices)
    if layer_indices:
        if max(layer_indices) - min(layer_indices) + 1 != n_layers:
            fail(f"non-contiguous layer indices {sorted(layer_indices)}; "
                 "mlx-lm can only target the last N blocks. Merge instead.")
        total_model_layers = peft_cfg.get("num_hidden_layers")
        if total_model_layers is not None:
            if max(layer_indices) != int(total_model_layers) - 1:
                fail(
                    f"adapter covers layers {sorted(layer_indices)} but the model has "
                    f"{total_model_layers} layers (indices 0-{int(total_model_layers) - 1}). "
                    "mlx-lm applies LoRA to the LAST num_layers blocks, so this adapter "
                    "would target the wrong layers. Retrain on the tail layers or merge instead."
                )
        else:
            msg = (
                "cannot determine total model layer count (num_hidden_layers not in "
                "adapter_config.json), so tail-layer alignment cannot be verified. "
                "mlx-lm applies LoRA to the LAST num_layers blocks; pass "
                "--allow-unknown-layer-count only after verifying this adapter covers "
                "the base model's tail layers."
            )
            if not allow_unknown_layer_count:
                fail(msg)
            print(f"WARNING: {msg}", file=sys.stderr)

    rank = peft_cfg["r"]
    # rsLoRA scales by alpha/√r; plain LoRA by alpha/r. With a uniform rank
    # (rank_pattern is refused above) this is a single constant either way,
    # which mlx-lm's lora_parameters.scale represents exactly.
    use_rslora = bool(peft_cfg.get("use_rslora"))
    scale = peft_cfg["lora_alpha"] / (rank ** 0.5 if use_rslora else rank)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(mlx_weights, str(out_dir / "adapters.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": n_layers,
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "dropout": 0.0,           # inference-irrelevant
            "keys": sorted(rel_keys),
        },
        "_converted_from": "peft",
        "_peft_lora_alpha": peft_cfg["lora_alpha"],
        "_peft_use_rslora": use_rslora,
        "_source": str(in_dir),
    }, indent=2))

    print(f"OK: {len(mlx_weights)} tensors -> {out_dir}/adapters.safetensors")
    print(f"    rank={rank} scale={scale} num_layers={n_layers}")
    print(f"    keys={sorted(rel_keys)}")
    print("NEXT: run the temp-0 parity check in the module docstring before shipping.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", required=True,
                   help="PEFT adapter dir (e.g. lora/example-local/final)")
    p.add_argument("--out", dest="out_dir", required=True,
                   help="Output dir for mlx-lm adapter")
    p.add_argument("--allow-unknown-layer-count", action="store_true",
                   help="Allow conversion when adapter_config.json lacks num_hidden_layers")
    args = p.parse_args()
    convert(
        Path(args.in_dir).expanduser(),
        Path(args.out_dir).expanduser(),
        allow_unknown_layer_count=args.allow_unknown_layer_count,
    )


if __name__ == "__main__":
    main()
