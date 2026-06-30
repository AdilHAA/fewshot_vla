"""Trajectory/video-conditioned Hyper-LoRA on frozen SmolVLA.

Importing registers the `traj_hyper_lora_smolvla` policy type for the stock lerobot CLI,
mirroring src/hyper_lora/__init__.py. The existing hyper_lora_smolvla policy is not modified.
"""

from .configuration_traj_hyper_lora_smolvla import TrajHyperLoRASmolVLAConfig
from .fusion_hypernetwork import FusionHyperNetwork
from .modeling_traj_hyper_lora_smolvla import TrajHyperLoRASmolVLAPolicy

__all__ = [
    "TrajHyperLoRASmolVLAConfig",
    "TrajHyperLoRASmolVLAPolicy",
    "FusionHyperNetwork",
]
