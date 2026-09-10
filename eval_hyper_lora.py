"""Eval our hypernetwork policy via stock `lerobot-eval`.

`--policy.path=<our_ckpt>` resolves to HyperLoRASmolVLAPolicy.

    python eval_hyper_lora.py \
        --policy.path=outputs/hyper_lora_run/checkpoints/last/pretrained_model \
        --env.type=libero --env.task=libero_object \
        --eval.n_episodes=10 --eval.batch_size=10 \
        --policy.device=cuda --policy.use_amp=false
"""

import os

# Set before mujoco is imported. See eval_libero_pro.py for the device pin.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import src.hyper_lora  # noqa: E402,F401 — registers HyperLoRASmolVLAPolicy
import src.hyper_lora_traj  # noqa: E402,F401 — registers TrajHyperLoRASmolVLAPolicy
import src.libero_pro  # noqa: E402,F401 — registers LIBERO-Pro perturbed suites
from lerobot.scripts.lerobot_eval import eval_main  # noqa: E402


def _patch_eval_image_flip() -> None:
    """EVAL_FLIP_LR="<image key> [<image key> ...]": mirror those env frames
    left-right before they reach the policy. Diagnostic for a dataset whose frames
    are horizontally mirrored w.r.t. the simulator render (yzembodied LIBERO-90:
    the scene's fixed cabinet sits on the opposite side in dataset vs env frames).
    A policy trained on mirrored frames + real actions is consistent in ITS world;
    mirroring the env frames puts it back in-distribution without retraining.
    Off (unset) = byte-identical behaviour."""
    keys = [k for k in os.environ.get("EVAL_FLIP_LR", "").replace(",", " ").split() if k]
    if not keys:
        return
    import torch
    import lerobot.envs.utils as _eu
    import lerobot.scripts.lerobot_eval as _le

    orig = _eu.preprocess_observation

    def flipped(observation, *args, **kwargs):
        out = orig(observation, *args, **kwargs)
        for k in keys:
            if k in out:
                out[k] = torch.flip(out[k], dims=[-1])   # (B, C, H, W): flip W
        return out

    _eu.preprocess_observation = flipped
    if hasattr(_le, "preprocess_observation"):        # bound by name at import
        _le.preprocess_observation = flipped
    print(f"[eval] EVAL_FLIP_LR: mirroring {keys} left-right")


if __name__ == "__main__":
    _patch_eval_image_flip()
    eval_main()
