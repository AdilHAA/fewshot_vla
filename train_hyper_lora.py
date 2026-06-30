"""Train the hypernetwork online via stock `lerobot-train`.

The LIBERO-finetuned base is frozen end-to-end; only the hypernet trains, via
the flow-matching loss on all `lerobot/libero` tasks. Generalization is measured
on LIBERO-Pro, not here.

    python train_hyper_lora.py \
        --policy.type=hyper_lora_smolvla \
        --dataset.repo_id=lerobot/libero \
        --dataset.use_imagenet_stats=false \
        --policy.push_to_hub=false \
        --steps=100000 --batch_size=8 \
        --output_dir=outputs/hyper_lora_run

`smolvla_libero` was trained with a non-default narrow action expert
(expert_width_multiplier=0.5). We derive those SmolVLA fields from the base
checkpoint's config.json and inject them so the skeleton matches the base
state_dict. Explicit CLI flags win; resuming a checkpoint skips injection.
"""

import sys

import src.hyper_lora  # noqa: F401 — registers the hyper_lora_smolvla policy type
import src.hyper_lora_traj  # noqa: F401 — registers the traj_hyper_lora_smolvla policy type
from src.hyper_lora import HyperLoRASmolVLAConfig
from src.hyper_lora.base_config import base_config_overrides
from lerobot.scripts.lerobot_train import main


def _inject_base_config_overrides() -> None:
    argv = sys.argv[1:]
    # Both policy types load the same frozen base, so both need the narrow-expert
    # SmolVLA config overrides derived from the base checkpoint.
    if not any(
        f"--policy.type={t}" in argv
        for t in ("hyper_lora_smolvla", "traj_hyper_lora_smolvla")
    ):
        return
    if any(a.startswith("--policy.path=") for a in argv):
        return  # resuming our own checkpoint: its config.json is authoritative

    base = HyperLoRASmolVLAConfig.base_smolvla_path
    for a in argv:
        if a.startswith("--policy.base_smolvla_path="):
            base = a.split("=", 1)[1]
    if not base or str(base).lower() == "none":
        return

    extra = base_config_overrides(str(base), argv)
    if extra:
        print(f"[train_hyper_lora] matching SmolVLA config to base {base!r}: "
              f"{len(extra)} override(s)")
        sys.argv += extra


if __name__ == "__main__":
    _inject_base_config_overrides()
    main()
