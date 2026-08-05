"""Печатает кондишенинг-настройки обученного прогона из его train_config.json.
Ровно те поля, которые обязаны совпасть у нового арма с его компаратором.

  python scripts/show_run_config.py outputs/*/checkpoints/last/pretrained_model
"""
import json, os, sys

KEYS = ["hn_traj_encoder", "hn_p_self", "hn_context_k", "hn_vjepa_grid",
        "hn_dino_grid", "hn_dino_n_reg", "hn_traj_pos_emb", "hn_dino_tokens",
        "hn_use_vlm_vision", "hn_use_dino", "hn_xpair_cache_path",
        "hn_frame_bank_path", "train_action_expert"]

for path in sys.argv[1:]:
    cfg = path if path.endswith(".json") else os.path.join(path, "train_config.json")
    if not os.path.isfile(cfg):
        print(f"{path}: нет train_config.json"); continue
    d = json.load(open(cfg))
    pol = d.get("policy", d)
    print(f"\n=== {path}")
    print(f"  batch={d.get('batch_size')} steps={d.get('steps')} seed={d.get('seed')} "
          f"aug={(d.get('dataset') or {}).get('image_transforms', {}).get('enable')}")
    for k in KEYS:
        if k in pol:
            print(f"  {k:22s} {pol[k]}")
