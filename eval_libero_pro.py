"""Register LIBERO-Pro perturbed suites, then hand off to stock `lerobot-eval`.

    python eval_libero_pro.py \
        --policy.path=HuggingFaceVLA/smolvla_libero \
        --env.type=libero --env.task=libero_object_swap \
        --eval.n_episodes=10 --eval.batch_size=10 \
        --policy.device=cuda --policy.use_amp=false

Axes: libero_<base>_{lan|object|swap|task} (Semantic/Object/Position/Task).
Report per-axis, never averaged.
"""

import os

# Set before mujoco is imported (via src.libero_pro). Plain egl enumerates all
# EGL devices and may pick a non-renderable one; pin the GPU device.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import src.libero_pro  # noqa: E402,F401 — registers LIBERO-Pro suites on import
from lerobot.scripts.lerobot_eval import eval_main  # noqa: E402

if __name__ == "__main__":
    eval_main()
