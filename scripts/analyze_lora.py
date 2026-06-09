"""Analysis: do the generated LoRAs actually depend on the task?

For each task instruction, runs the trained hypernetwork (text conditioning
only — vision streams are omitted, which for vision-conditioned checkpoints probes just the
text pathway) and dumps:

  <out>/lora_norms.csv    per-task, per-(module, layer) Frobenius norm of
                          ΔW = scaling * W_up @ W_down, plus the total norm.
                          Total ≈ 0 for every task => the HN degenerated to a
                          no-op; uniform across tasks => it ignores the input.
  <out>/cosine_sim.csv    pairwise cosine similarity between the flattened
                          generated (W_down, W_up) fingerprints. Structure
                          (clusters by object/suite) is direct evidence
                          that the HN conditions on the instruction.
  <out>/fingerprints.npz  raw fingerprint vectors + task strings, for
                          t-SNE / clustering figures.

Usage (from repo root, venv active):
    python scripts/analyze_lora.py --ckpt outputs/hyper_lora_vision/checkpoints/last/pretrained_model
    python scripts/analyze_lora.py --ckpt ... --tasks-file my_instructions.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.hyper_lora  # noqa: F401  (registers the policy type)
from src.hyper_lora.modeling_hyper_lora_smolvla import HyperLoRASmolVLAPolicy


def load_task_instructions(repo_id: str) -> list[str]:
    """Task strings from a lerobot dataset's metadata (no data download)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    tasks = LeRobotDatasetMetadata(repo_id).tasks
    # tasks is a DataFrame (index = task string) in lerobot>=0.4, a dict in
    # older versions — normalize both.
    if hasattr(tasks, "index"):
        return [str(t) for t in tasks.index]
    return [str(v) for v in tasks.values()]


@torch.no_grad()
def generate_lora(policy: HyperLoRASmolVLAPolicy, instructions: list[str], device: str):
    """Run the HN per instruction; return (norms, fingerprints).

    norms:        {task: {f"{mod}.{layer}": float}}
    fingerprints: (N_tasks, D) float32 — flattened (W_down, W_up) per task.
    """
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    max_len = policy.config.tokenizer_max_length

    norms: dict[str, dict[str, float]] = {}
    fingerprints: list[np.ndarray] = []
    scaling = policy.config.lora_alpha / policy.config.lora_rank

    for task in instructions:
        enc = tokenizer(
            task, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_len,
        )
        lang_tokens = enc["input_ids"].to(device)
        lang_masks = enc["attention_mask"].to(device)
        text_embeds = policy._embed_language(lang_tokens)

        weights = policy.hypernet(text_embeds, lang_masks, None, None)

        task_norms: dict[str, float] = {}
        flat_parts: list[np.ndarray] = []
        for mod, layers in weights.items():
            for layer_idx, (w_d, w_u) in layers.items():
                w_d32 = w_d[0].float()  # (r, in)
                w_u32 = w_u[0].float()  # (out, r)
                delta = scaling * (w_u32 @ w_d32)
                task_norms[f"{mod}.{layer_idx}"] = float(delta.norm().cpu())
                flat_parts.append(w_d32.flatten().cpu().numpy())
                flat_parts.append(w_u32.flatten().cpu().numpy())
        norms[task] = task_norms
        fingerprints.append(np.concatenate(flat_parts).astype(np.float32))

    return norms, np.stack(fingerprints)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="pretrained_model dir of a trained run")
    parser.add_argument("--repo-id", default="lerobot/libero", help="dataset to take task strings from")
    parser.add_argument("--tasks-file", default=None, help="txt file, one instruction per line (overrides --repo-id)")
    parser.add_argument("--out", default=None, help="output dir (default: <ckpt>/../../analysis)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(args.ckpt).resolve().parent.parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    if args.tasks_file:
        instructions = [l.strip() for l in Path(args.tasks_file).read_text().splitlines() if l.strip()]
    else:
        instructions = load_task_instructions(args.repo_id)
    print(f"{len(instructions)} task instructions")

    policy = HyperLoRASmolVLAPolicy.from_pretrained(args.ckpt)
    if policy.config.static_lora:
        raise SystemExit("static_lora checkpoint: the LoRA is constant by construction, nothing to analyze.")
    policy.to(args.device).eval()

    norms, fingerprints = generate_lora(policy, instructions, args.device)

    # --- lora_norms.csv ---
    cols = sorted(next(iter(norms.values())).keys(), key=lambda c: (c.split(".")[0], int(c.split(".")[1])))
    with open(out / "lora_norms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "total"] + cols)
        for task in instructions:
            row_norms = norms[task]
            total = float(np.sqrt(sum(v**2 for v in row_norms.values())))
            w.writerow([task, f"{total:.4f}"] + [f"{row_norms[c]:.4f}" for c in cols])

    # --- cosine_sim.csv ---
    fp = fingerprints / np.clip(np.linalg.norm(fingerprints, axis=1, keepdims=True), 1e-8, None)
    sim = fp @ fp.T
    with open(out / "cosine_sim.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + instructions)
        for task, row in zip(instructions, sim):
            w.writerow([task] + [f"{v:.4f}" for v in row])

    np.savez_compressed(out / "fingerprints.npz", fingerprints=fingerprints, tasks=np.array(instructions))

    totals = [float(np.sqrt(sum(v**2 for v in norms[t].values()))) for t in instructions]
    off_diag = sim[~np.eye(len(sim), dtype=bool)]
    print(f"ΔW total norm: min={min(totals):.4f} max={max(totals):.4f} mean={float(np.mean(totals)):.4f}")
    print(f"cross-task cosine: min={off_diag.min():.4f} max={off_diag.max():.4f} mean={off_diag.mean():.4f}")
    print(f"wrote {out}/lora_norms.csv, cosine_sim.csv, fingerprints.npz")
    if max(totals) < 1e-3:
        print("WARNING: all ΔW ≈ 0 — the hypernetwork degenerated to a no-op.")
    elif off_diag.mean() > 0.999:
        print("WARNING: fingerprints are ~identical across tasks — the HN ignores its input.")


if __name__ == "__main__":
    main()
