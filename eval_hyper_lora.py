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

if __name__ == "__main__":
    eval_main()
