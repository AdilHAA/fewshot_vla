"""Local copy of `lerobot/smolvla_base` with LIBERO I/O features (T3 step 1).

Why this exists: lerobot's make_policy KEEPS a checkpoint's input_features when
they are non-empty, and smolvla_base carries the SO-100 ones (camera1/2/3,
state[6], action[6]) — training it on LIBERO data directly would look up batch
keys that don't exist. This script snapshots the checkpoint and rewrites ONLY
config.json:

  input_features  -> observation.images.image / image2 [3,256,256], observation.state [8]
  output_features -> action [7]
  n_action_steps  -> 1   (the paper's sim-eval protocol; baked into the config so
                          every later eval and the HN stage inherit it)

Why the WEIGHTS still load unchanged: SmolVLA pads state/action to
max_state_dim=max_action_dim=32, so the projections are feature-dim-agnostic,
and the vision tower is shared across cameras (no per-camera weights). The
checkpoint's preprocessor stats (dim 6) are IGNORED at train time — lerobot-train
overrides normalizer/unnormalizer stats with the training dataset's stats when
--policy.path is set — and the finetuned checkpoint saves the new stats.

  python scripts/prepare_base_smolvla_libero.py --out outputs/base/smolvla_base_libero_io
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="lerobot/smolvla_base",
                   help="HF repo id or a local checkpoint dir")
    p.add_argument("--out", default="outputs/base/smolvla_base_libero_io")
    args = p.parse_args(argv)

    src = Path(args.src)
    if not src.is_dir():
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(args.src))
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists — remove it or pick another --out")
    shutil.copytree(src, out)

    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    old = {k: cfg.get(k) for k in ("input_features", "output_features", "n_action_steps")}
    cfg["input_features"] = {
        "observation.images.image":  {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
        "observation.state":         {"type": "STATE",  "shape": [8]},
    }
    cfg["output_features"] = {"action": {"type": "ACTION", "shape": [7]}}
    cfg["n_action_steps"] = 1
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"copied {src} -> {out}")
    print(f"config.json rewritten: {json.dumps(old)}")
    print("  -> LIBERO features (image/image2 [3,256,256], state[8], action[7]), "
          "n_action_steps=1")
    for k in ("vlm_model_name", "num_vlm_layers", "expert_width_multiplier",
              "chunk_size", "resize_imgs_with_padding", "pad_language_to"):
        print(f"  kept {k} = {cfg.get(k)}")


if __name__ == "__main__":
    main()
