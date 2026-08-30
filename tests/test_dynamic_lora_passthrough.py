"""DynamicLoRALinear attribute passthrough (T8). Pure torch, runs anywhere.
Run: python3 tests/test_dynamic_lora_passthrough.py
"""
import os
import sys
import types

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Stub the package chain so importing the module doesn't pull src/hyper_lora's
# __init__ (which imports lerobot) — same trick as the other headless tests.
for pkgname, rel in [("src", "src"), ("src.hyper_lora", "src/hyper_lora")]:
    if pkgname not in sys.modules:
        pkg = types.ModuleType(pkgname)
        pkg.__path__ = [os.path.join(ROOT, rel)]
        sys.modules[pkgname] = pkg

from src.hyper_lora.dynamic_lora import DynamicLoRALinear


def test_weight_bias_passthrough():
    """smolvlm_with_expert reads `.weight.dtype` (and dtype-casts via `.weight`)
    on the expert/VLM attention projections; a wrapper without the attribute
    crashes there. The passthrough must expose the FROZEN base tensors."""
    base = nn.Linear(8, 4, bias=True).to(torch.float16)
    w = DynamicLoRALinear(base, lora_rank=2, lora_alpha=4)
    assert w.weight is base.weight and w.bias is base.bias
    assert w.weight.dtype == torch.float16
    assert not w.weight.requires_grad          # frozen by the wrapper's __init__
    nb = nn.Linear(8, 4, bias=False)
    assert DynamicLoRALinear(nb).bias is None


def test_passthrough_changes_no_behaviour():
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    w = DynamicLoRALinear(base, lora_rank=2, lora_alpha=4)
    x = torch.randn(3, 5, 8)
    assert torch.equal(w(x), base(x))          # no LoRA set -> identical
    wd = torch.randn(3, 2, 8)
    wu = torch.zeros(3, 4, 2)
    w.set_lora_weights(wd, wu)
    assert torch.allclose(w(x), base(x))       # zero W_up -> still identical
    # weight is a property, not a registered parameter: no duplicate in state_dict
    assert set(w.state_dict()) == {"base_layer.weight", "base_layer.bias"}


RUN = [test_weight_bias_passthrough, test_passthrough_changes_no_behaviour]
if __name__ == "__main__":
    for fn in RUN:
        fn(); print(f"PASS {fn.__name__}")
    print("ALL DYNAMIC-LORA TESTS PASS")
