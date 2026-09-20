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
from lerobot.scripts import lerobot_eval  # noqa: E402

_add_envs_task = lerobot_eval.add_envs_task


def add_envs_task(env, observation):
    """lerobot's helper only fills observation['task'] (the instruction). Two LIBERO
    tasks in different scenes can share an instruction, so the hypernetwork resolves
    its demo context by the bddl stem instead: LiberoEnv.task, passed on as 'subtask'
    (a key lerobot's processor pipeline forwards to the policy batch)."""
    observation = _add_envs_task(env, observation)
    if hasattr(env.envs[0], "task"):
        observation["subtask"] = list(env.call("task"))
    return observation


# rollout() imported the name INTO lerobot_eval, so the rebind must land there
# (patching lerobot.envs.utils would be looked up by nobody).
lerobot_eval.add_envs_task = add_envs_task

if __name__ == "__main__":
    lerobot_eval.eval_main()
