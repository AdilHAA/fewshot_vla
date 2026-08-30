"""hn_lora_target=expert_mlp (T8) — config validation + (GPU-gated) site patching.
Needs lerobot; the policy-construction test additionally downloads the VLM, so it
only runs with HN_GPU_TESTS=1 (GPU machine).
Run: python3 tests/test_expert_lora_target.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import lerobot  # noqa: F401
except Exception:
    print("SKIP: lerobot not installed (GPU-machine test)")
    sys.exit(0)

from src.hyper_lora.configuration_hyper_lora_smolvla import HyperLoRASmolVLAConfig
from src.hyper_lora_traj.configuration_traj_hyper_lora_smolvla import (
    TrajHyperLoRASmolVLAConfig,
)


def test_config_validation():
    HyperLoRASmolVLAConfig(hn_lora_target="expert_mlp")            # frozen: fine
    try:
        HyperLoRASmolVLAConfig(hn_lora_target="expert_mlp", train_action_expert=True)
        raise SystemExit("expert site + trainable expert must be rejected")
    except ValueError:
        pass
    try:
        HyperLoRASmolVLAConfig(hn_lora_target="expert_attention")
        raise SystemExit("unknown target must be rejected")
    except ValueError:
        pass
    try:
        TrajHyperLoRASmolVLAConfig(hn_lora_target="expert_mlp", hn_inject_vlm_kv=True)
        raise SystemExit("expert site + vlm kv site must be rejected")
    except ValueError:
        pass
    TrajHyperLoRASmolVLAConfig(hn_lora_target="expert_mlp")        # defaults: fine


def test_expert_site_patching():
    """Build the real policy (downloads the VLM checkpoint): the wrappers must sit
    on lm_expert, the expert must be frozen, and the HN heads must match the
    expert's MLP dims."""
    if os.environ.get("HN_GPU_TESTS") != "1":
        print("  (skipped policy construction — set HN_GPU_TESTS=1 on the GPU machine)")
        return
    from src.hyper_lora.dynamic_lora import DynamicLoRALinear
    from src.hyper_lora.modeling_hyper_lora_smolvla import HyperLoRASmolVLAPolicy

    cfg = HyperLoRASmolVLAConfig(hn_lora_target="expert_mlp", base_smolvla_path=None)
    policy = HyperLoRASmolVLAPolicy(cfg)
    expert = policy.model.vlm_with_expert.lm_expert
    n_layers = len(expert.layers)
    for mod in ("gate_proj", "up_proj", "down_proj"):
        assert set(policy._patched[mod]) == set(range(n_layers))
        for i in range(n_layers):
            assert isinstance(getattr(expert.layers[i].mlp, mod), DynamicLoRALinear)
    # HN heads sized by the EXPERT dims (hidden/intermediate), not the VLM's
    h, inter = expert.config.hidden_size, expert.config.intermediate_size
    assert policy.hypernet.target_modules == {
        "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h)}
    # the VLM MLP is untouched
    vlm_mlp = policy._vlm_text_model().layers[0].mlp
    assert not isinstance(vlm_mlp.gate_proj, DynamicLoRALinear)
    counts = policy.trainable_parameter_count()
    assert counts["lm_expert"] == 0 and counts["vlm"] == 0 and counts["hypernet"] > 0
    assert policy.hypernet.num_layers == n_layers


RUN = [test_config_validation, test_expert_site_patching]
if __name__ == "__main__":
    for fn in RUN:
        fn(); print(f"PASS {fn.__name__}")
    print("ALL EXPERT-TARGET TESTS PASS")
